import re
from copy import copy
from typing import Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from preprocessing.prosqa import DAG


def evaluate_generation(
    parallel_model,
    valid_gen_dataloader,
    tokenizer,
    answers_val,
    cot_val,
    question_val,
    configs,
    max_new_tokens: int,
    scheduled_stage: int,
    rank: int,
    wandb_run=None,
) -> Tuple[int, int, int, int, int]:
    """
    Run validation-time generation and compute:
      - best-of-N answer accuracy
      - best-of-N CoT exact match
      - greedy best-of-1 answer accuracy
      - greedy best-of-1 CoT exact match

    Returns (cor, cor_cot, cor_1, cor_cot_1, total), all as Python ints
    AFTER distributed reduction.
    """

    def _eval_pass(
        eval_best_of: int,
        decoding_mode: str,
    ) -> Tuple[int, int, int]:
        """
        Single evaluation pass:
          - eval_best_of: number of samples per example
          - decoding_mode: 'greedy' or 'config'
        """
        total_length = len(valid_gen_dataloader)

        desc = (
            "Test Accuracy (greedy)"
            if decoding_mode == "greedy"
            else "Test Accuracy (best-of-N)"
        )
        pbar = tqdm(
            colour="blue", desc=desc, total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
        )

        with torch.no_grad():
            parallel_model.module.eval()
            example_counter = 0
            for batch_idx, batch in enumerate(valid_gen_dataloader):
                # Keep indices and graphs on CPU; move only tensor inputs to device
                batch_idx_tensor = batch["idx"]
                batch_graphs = batch["graph"]

                model_batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v is not None and k not in ["idx", "graph", "position_ids"]
                }

                if isinstance(model_batch["input_ids"], torch.Tensor):
                    batch_size = model_batch["input_ids"].shape[0]
                else:
                    batch_size = len(model_batch["input_ids"])

                # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                generate_kwargs = dict(
                    **model_batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=not configs.only_eval,
                    pad_token_id=tokenizer.eos_token_id
                )

                if decoding_mode == "config" and eval_best_of > 1:
                    # best-of-N decoding using configured strategy
                    generate_kwargs["num_return_sequences"] = eval_best_of

                    sampling_strategy = getattr(
                        configs, "eval_sampling_strategy", "sample"
                    )

                    if sampling_strategy == "beam":
                        # use beam search; ensure num_beams >= num_return_sequences
                        num_beams = getattr(configs, "eval_num_beams", eval_best_of)
                        num_beams = max(num_beams, eval_best_of)
                        generate_kwargs["num_beams"] = num_beams
                        generate_kwargs["do_sample"] = False
                    elif sampling_strategy == "greedy":
                        # approximate greedy best-of-N via sampling (since true greedy
                        # does not support num_return_sequences > 1)
                        generate_kwargs.setdefault("do_sample", True)
                        generate_kwargs.setdefault(
                            "temperature", getattr(configs, "eval_temperature", 0.7)
                        )
                        generate_kwargs.setdefault(
                            "top_p", getattr(configs, "eval_top_p", 0.9)
                        )
                    else:
                        # default: sampling-based best-of-N (recommended)
                        generate_kwargs.setdefault("do_sample", True)
                        generate_kwargs.setdefault(
                            "temperature", getattr(configs, "eval_temperature", 0.7)
                        )
                        generate_kwargs.setdefault(
                            "top_p", getattr(configs, "eval_top_p", 0.9)
                        )
                elif decoding_mode == "greedy":
                    # explicit greedy: single sequence per example, no sampling
                    generate_kwargs["do_sample"] = False
                    generate_kwargs.pop("num_beams", None)
                    # do NOT set num_return_sequences here; we want exactly 1

                outputs = parallel_model.module.generate(**generate_kwargs)

                # HF generate may return a tensor or an object with .sequences
                if isinstance(outputs, torch.Tensor):
                    gen_seqs = outputs
                elif hasattr(outputs, "sequences"):
                    gen_seqs = outputs.sequences
                else:
                    gen_seqs = outputs[0]

                # Shape handling:
                # - For best-of-1, we want shape [batch_size, 1, seq_len]
                # - For best-of-N, HF returns [batch_size * N, seq_len]; reshape to [batch_size, N, seq_len]
                if eval_best_of > 1:
                    gen_seqs = gen_seqs.view(batch_size, eval_best_of, -1)
                else:
                    gen_seqs = gen_seqs.unsqueeze(1)

                total += batch_size
                pbar.update(batch_size)

                for i in range(batch_size):
                    # Retrieve original sample index
                    if isinstance(batch_idx_tensor, torch.Tensor):
                        test_idx = batch_idx_tensor[i].item()
                    else:
                        test_idx = batch_idx_tensor[i]

                    answer = answers_val[test_idx]
                    answer_cot = cot_val[test_idx]
                    question = question_val[test_idx]

                    # Per-example correctness across eval_best_of generations
                    answer_correct = False
                    cot_correct = False

                    # extract the related graph data (shared across samples)
                    graph_data = batch_graphs[i]
                    symbol_to_idx = {
                        symbol: j for j, symbol in enumerate(graph_data["idx_to_symbol"])
                    }
                    nodes = list(range(len(graph_data["idx_to_symbol"])))
                    edges = [
                        [edge[-1] for edge in graph_data["edges"] if edge[0] == node]
                        for node in nodes
                    ]
                    graph = DAG(nodes, [-1 for _ in nodes], edges)
                    paths = graph.get_paths_between(
                        graph_data["root"], graph_data["target"]
                    )
                    paths = [
                        [(path[k - 1], path[k]) for k in range(1, len(path))]
                        for path in paths
                    ]

                    eval_with_visible_steps = getattr(
                        configs, "eval_with_visible_steps", False
                    )
                    is_reversed = getattr(configs, "reversed", False)
                    total_steps = len(answer_cot.split("\n"))
                    n_abstracted_steps = min(scheduled_stage, total_steps)
                    n_visible_steps = total_steps - n_abstracted_steps

                    for sample_id in range(eval_best_of):
                        seq = gen_seqs[i, sample_id]
                        text_output = tokenizer.decode(
                            seq, skip_special_tokens=True
                        )
                        answer_output = (
                            text_output.split("#")[-1].replace(",", "").strip()
                        )
                        cot_output = (
                            ("\n".join(text_output.split("\n")[1:]))
                            .split("#")[0]
                            .strip()
                        )

                        if answer_output == answer:
                            answer_correct = True

                        # Print qualitative examples only for the greedy pass to avoid
                        # confusion with stochastic best-of-N decoding.
                        if (
                            decoding_mode == "greedy"
                            and example_counter < 5
                            and rank == 0
                            and sample_id == 0
                        ):
                            print(
                                f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                            )
                            print(f"Full output: '{tokenizer.decode(seq)}'")
                            print(f"Extracted Output: '{answer_output}'")

                        # === compute the correctness of cots for this sample ===
                        matches = [step.strip() for step in cot_output.split("\n")]
                        for pattern in configs.cot_patterns:
                            matches = [
                                (re.search(pattern, match) or match)
                                if isinstance(match, str)
                                else match
                                for match in matches
                            ]
                        matches_x = [
                            -1
                            if isinstance(match, str)
                            or not (match.group("x") in symbol_to_idx)
                            else symbol_to_idx[match.group("x")]
                            for match in matches
                        ]
                        matches_y = [
                            -1
                            if isinstance(match, str)
                            or not (match.group("y") in symbol_to_idx)
                            else symbol_to_idx[match.group("y")]
                            for match in matches
                        ]
                        solution = [
                            (match_x, match_y)
                            for (match_x, match_y) in zip(matches_x, matches_y)
                        ]

                        # check cot correctness
                        if eval_with_visible_steps:
                            # only validate visible steps
                            if n_visible_steps > 0:
                                if is_reversed:
                                    # visible steps are the first n_visible_steps
                                    visible_solution = solution[:n_visible_steps]
                                    if any(
                                        path[:n_visible_steps] == visible_solution
                                        for path in paths
                                    ):
                                        cot_correct = True

                                    if (
                                        decoding_mode == "greedy"
                                        and example_counter < 5
                                        and rank == 0
                                        and sample_id == 0
                                    ):
                                        print(f"Symbol to idx map: {symbol_to_idx}")
                                        print(
                                            f"Visible solution: '{visible_solution}'"
                                        )
                                        print(
                                            f"Correct traces: '{[path[:n_visible_steps] for path in paths]}'"
                                        )
                                else:
                                    # Visible steps are the last n_visible_steps
                                    visible_solution = solution[-n_visible_steps:]
                                    if any(
                                        path[-n_visible_steps:] == visible_solution
                                        for path in paths
                                    ):
                                        cot_correct = True

                                    if (
                                        decoding_mode == "greedy"
                                        and example_counter < 5
                                        and rank == 0
                                        and sample_id == 0
                                    ):
                                        print(f"Symbol to idx map: {symbol_to_idx}")
                                        print(
                                            f"Visible solution: '{visible_solution}'"
                                        )
                                        print(
                                            f"Correct traces: '{[path[-n_visible_steps:] for path in paths]}'"
                                        )
                            else:
                                # All steps are abstracted, just check the answer
                                cot_correct = True
                        else:
                            if solution in paths:
                                cot_correct = True

                            # print some examples
                            if (
                                decoding_mode == "greedy"
                                and example_counter < 5
                                and rank == 0
                                and sample_id == 0
                            ):                                
                                print(f"Symbol to idx map: {symbol_to_idx}")
                                print(f"Visible solution: '{solution}'")
                                print(f"Correct traces: '{paths}'")

                        # Early exit if both answer and CoT are already correct
                        if answer_correct and cot_correct:
                            break

                    cor += answer_correct
                    cor_cot += cot_correct
                    example_counter += 1

                pbar.set_description(
                    f"{desc}: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        return cor.item(), cor_cot.item(), total.item()

    # --- orchestrate passes ---
    eval_best_of_cfg = getattr(configs, "eval_best_of", 1)

    # 1) Greedy best-of-1 pass (always)
    cor_1, cor_cot_1, total = _eval_pass(eval_best_of=1, decoding_mode="greedy")

    # 2) Best-of-N pass (if N > 1); otherwise reuse greedy numbers
    if eval_best_of_cfg > 1:
        cor, cor_cot, _ = _eval_pass(
            eval_best_of=eval_best_of_cfg, decoding_mode="config"
        )
    else:
        cor, cor_cot = cor_1, cor_cot_1

    if rank == 0:
        print(f"(best-of-N) Accuracy on validation set: {cor} / {total} = {cor/total}")
        print(f"(best-of-N) CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        print(
            f"(Greedy best-of-1) Accuracy on validation set: {cor_1} / {total} = {cor_1/total}"
        )
        print(
            f"(Greedy best-of-1) CoT match on validation set: {cor_cot_1} / {total} = {cor_cot_1/total}"
        )

    if wandb_run:
        log_dict = {
            "eval/acc": cor / total,
            "eval/cot_em": cor_cot / total,
            "eval/acc_1": cor_1 / total,
            "eval/cot_em_1": cor_cot_1 / total,
            "eval/scheduled_stage": scheduled_stage,
        }

        wandb_run.log(log_dict)

    return cor, cor_cot, cor_1, cor_cot_1, total


