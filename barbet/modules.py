import gc
from pathlib import Path
from torchapp.modules import GeneralLightningModule
# import pandas as pd
import polars as pl
import torch
from collections import defaultdict
from hierarchicalsoftmax.inference import (
    greedy_lineage_probabilities,
    node_probabilities,
    greedy_predictions,
)
from barbet.data import RANKS


class BarbetLightningModule(GeneralLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_prediction(
        self,
        barbet,
        names: list[str] | str,
        threshold: float = 0.0,
        save_probabilities: bool = False,
        save_context_vectors: bool = False,
    ):
        self.names = names
        self.classification_tree = self.hparams.classification_tree
        # self.logits = defaultdict(lambda: 0.0)
        # self.counts = defaultdict(lambda: 0)
        self.counter = 0
        unique_names = list(dict.fromkeys(names)) if isinstance(names, list) else [names]
        genome_count = len(unique_names)
        self.name_to_index = {name: i for i, name in enumerate(unique_names)}
        self.logits = torch.zeros(
            (genome_count, self.classification_tree.layer_size),
            dtype=torch.float16, 
        )
        self.counts = torch.zeros(
            (genome_count,) ,
            dtype=torch.int32, 
        )
        self.category_names = [
            barbet.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root
        ]
        self.barbet = barbet
        self.threshold = threshold
        self.save_probabilities = save_probabilities
        self.save_context_vectors = save_context_vectors
        self.context_vectors = None

    def on_predict_batch_end(self, results, batch, batch_idx, dataloader_idx=0):
        batch_size = len(results)
        save_ctx = getattr(self, "save_context_vectors", False) and hasattr(self.model, "last_context_vector")
        if save_ctx:
            ctx_batch = self.model.last_context_vector.float().cpu()
            if self.context_vectors is None:
                self.context_vectors = torch.zeros(
                    (len(self.name_to_index), ctx_batch.shape[1]),
                    dtype=torch.float32,
                )

        if isinstance(self.names, str):
            genome_index = self.name_to_index[self.names]
            self.counts[genome_index] += batch_size
            self.logits[genome_index,:] += results.sum(dim=0).half().cpu()
            if save_ctx:
                self.context_vectors[genome_index, :] += ctx_batch.sum(dim=0)
        else:
            prev_name = self.names[self.counter]
            start_i = 0
            for end_i in range(batch_size):
                current_name = self.names[self.counter + end_i]
                if current_name != prev_name:
                    genome_index = self.name_to_index[prev_name]
                    self.counts[genome_index] += (end_i - start_i)
                    self.logits[genome_index,:] += results[start_i:end_i].sum(dim=0).half().cpu()
                    if save_ctx:
                        self.context_vectors[genome_index, :] += ctx_batch[start_i:end_i].sum(dim=0)
                    start_i = end_i
                    prev_name = current_name
            
            # Handle the last chunk
            assert start_i < batch_size, "Start index should be less than batch size"
            genome_index = self.name_to_index[prev_name]
            self.logits[genome_index,:] += results[start_i:].sum(dim=0).half().cpu()
            self.counts[genome_index] += (batch_size - start_i)
            if save_ctx:
                self.context_vectors[genome_index, :] += ctx_batch[start_i:].sum(dim=0)
            self.counter += batch_size

    def on_predict_epoch_end(self):
        print("Consolidating results per genome...")
        names = list(self.name_to_index.keys())
        self.logits /= self.counts.unsqueeze(1)  # Normalize logits by counts
        if getattr(self, "save_context_vectors", False) and getattr(self, "context_vectors", None) is not None:
            self.context_vectors /= self.counts.unsqueeze(1)  # Normalize context vectors by counts
        del self.counts
        gc.collect()

        # Prepare column names and initialize empty lists
        output_columns = ['name']
        new_cols = {}
        new_cols['name'] = names

        for rank in RANKS:
            pred_col = f"{rank}_prediction"
            prob_col = f"{rank}_probability"
            output_columns += [pred_col, prob_col]
            new_cols[pred_col] = []
            new_cols[prob_col] = []

        # Convert to probabilities
        if self.save_probabilities:
            print("Converting to probabilities...")
            probabilities = node_probabilities(
                self.logits, 
                root=self.classification_tree,
                progress_bar=True,
            )
            self.probabilities = probabilities

            del self.logits
            gc.collect()

            print("Saving in dataframe...")
            self.results_df = pl.DataFrame(
                data=probabilities,
                schema=self.category_names
            ).with_columns([
                pl.Series("name", names, dtype=pl.Utf8)
            ]).with_columns([
                pl.col("name").cast(pl.Utf8)
            ]).select(["name", *self.category_names])
            
            # get greedy predictions which can use the raw activation or the softmax probabilities
            print("Getting greedy predictions...")
            predictions = greedy_predictions(
                probabilities,
                root=self.classification_tree,
                threshold=self.threshold,
                progress_bar=True,
            )

            # Prepare essentials
            num_rows = self.results_df.height

            for i in range(num_rows):
                prediction_node = predictions[i]
                lineage = prediction_node.ancestors[1:] + (prediction_node,)
                probability = 1.0

                for rank, lineage_node in zip(RANKS, lineage):
                    node_name = self.barbet.node_to_str(lineage_node)
                    pred_col = f"{rank}_prediction"
                    prob_col = f"{rank}_probability"

                    new_cols[pred_col].append(node_name)

                    if node_name in self.results_df.columns:
                        probability = self.results_df[node_name][i]

                    new_cols[prob_col].append(probability)

            # Add new columns to the Polars DataFrame
            self.results_df = self.results_df.with_columns(
                [pl.Series(name, values) for name, values in new_cols.items()]
            )
            self.results_df = self.results_df[output_columns]
        else:
            print("Finding greedy predictions...")
            results = greedy_lineage_probabilities(
                self.logits, 
                root=self.classification_tree,
                threshold=self.threshold,
                progress_bar=True,
            )

            del self.logits
            gc.collect()

            for row in results:
                if not len(row) == len(RANKS):
                    breakpoint()
                assert len(row) == len(RANKS), f"Row length {len(row)} does not match number of ranks {len(RANKS)}"
                for rank_index, (node, probability) in enumerate(row):
                    rank = RANKS[rank_index]
                    node_name = self.barbet.node_to_str(node)
                    pred_col = f"{rank}_prediction"
                    prob_col = f"{rank}_probability"

                    new_cols[pred_col].append(node_name)
                    new_cols[prob_col].append(probability)

            # Create the DataFrame
            self.results_df = pl.DataFrame(
                data=new_cols,
                schema=output_columns
            ).with_columns([
                pl.col("name").cast(pl.Utf8)
            ]).select(output_columns)

    def save_context_vectors_tsv(self, output_path: str | Path) -> None:
        import gzip
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        names = list(self.name_to_index.keys())
        if getattr(self, "context_vectors", None) is None:
            raise RuntimeError("No context vectors available to save.")

        ctx_vecs = self.context_vectors.detach().cpu().numpy()

        is_gz = str(output_path).endswith(".gz")
        open_fn = gzip.open if is_gz else open

        with open_fn(output_path, "wt", encoding="utf-8") as f:
            f.write("name\tcontext_vector\n")
            for name, vec in zip(names, ctx_vecs):
                vec_str = ",".join(map(str, vec.tolist()))
                f.write(f"{name}\t{vec_str}\n")

    def generate_top_predictions(self, barbet, num_predictions: int = 1, global_predictions: bool = False) -> pl.DataFrame:
        tree = self.classification_tree
        nodes = [node for node in tree.node_list_softmax if not node.is_root]
        node_to_col_idx = {node: i for i, node in enumerate(nodes)}

        names = list(self.name_to_index.keys())
        if hasattr(self, "probabilities") and self.probabilities is not None:
            probs_matrix = self.probabilities
        elif hasattr(self, "logits") and self.logits is not None:
            probs_matrix = node_probabilities(self.logits, root=tree, progress_bar=False)
        else:
            raise RuntimeError("No probabilities or logits available to calculate top predictions.")

        num_predictions = max(1, num_predictions)
        rows = []

        if global_predictions:
            # Pre-group all softmax nodes by their depth in the tree.
            # Depth 1 = phylum, 2 = class, ..., 6 = species (for a 6-rank tree).
            nodes_by_depth: dict[int, list] = {}
            for node in nodes:
                nodes_by_depth.setdefault(node.depth, []).append(node)

            def get_joint_prob(node, genome_probs) -> float:
                """Return the absolute (joint) probability of any node.

                For nodes in the softmax layer this is read directly from
                probs_matrix.  For single-child nodes (absent from the softmax
                layer because no probability split is needed) the probability is
                inherited recursively from the parent.
                """
                if node.is_root:
                    return 1.0
                if node in node_to_col_idx:
                    return float(genome_probs[node_to_col_idx[node]])
                # Single-child node: probability equals parent's probability.
                return get_joint_prob(node.parent, genome_probs)

        for genome_idx, name in enumerate(names):
            genome_probs = probs_matrix[genome_idx]

            if global_predictions:
                # ── Step 1: walk the greedy conditional path ──────────────────
                # Identifies which node at each rank is in_predicted_lineage.
                # This mirrors the logic used in barbet-predictions.csv.
                greedy_node_at_rank: dict[int, object] = {}
                greedy_path_set: set = set()
                gp_parent = tree
                for rank_idx, rank in enumerate(RANKS):
                    gp_children = getattr(gp_parent, "children", [])
                    if not gp_children:
                        break
                    gp_valid = [c for c in gp_children if c in node_to_col_idx]
                    if not gp_valid:
                        # Single-child node: automatically selected.
                        gp_node = gp_children[0]
                    else:
                        gp_probs = [
                            (c, float(genome_probs[node_to_col_idx[c]]))
                            for c in gp_valid
                        ]
                        gp_probs.sort(key=lambda x: (-x[1], barbet.node_to_str(x[0])))
                        gp_node = gp_probs[0][0]
                    greedy_node_at_rank[rank_idx] = gp_node
                    greedy_path_set.add(gp_node)
                    gp_parent = gp_node

                # ── Step 2: emit global top-N for every rank ──────────────────
                for rank_idx, rank in enumerate(RANKS):
                    rank_depth = rank_idx + 1
                    greedy_node = greedy_node_at_rank.get(rank_idx)

                    # Single-child nodes are not in node_list_softmax so they
                    # cannot participate in a global ranking.  Emit them
                    # separately with inherited probability and local_prob=1.0.
                    is_single_child = (
                        greedy_node is not None
                        and greedy_node not in node_to_col_idx
                    )
                    if is_single_child:
                        joint_prob = get_joint_prob(greedy_node.parent, genome_probs)
                        rows.append({
                            "name": name,
                            "rank": rank,
                            "taxon": barbet.node_to_str(greedy_node),
                            "joint_probability": joint_prob,
                            "local_probability": 1.0,
                            "prediction_number": 1,
                            "in_predicted_lineage": True,
                        })
                        continue

                    rank_nodes = nodes_by_depth.get(rank_depth, [])
                    if not rank_nodes:
                        continue

                    # Sort globally by joint (absolute) probability.
                    node_joint = [
                        (n, float(genome_probs[node_to_col_idx[n]]))
                        for n in rank_nodes
                    ]
                    node_joint.sort(key=lambda x: (-x[1], barbet.node_to_str(x[0])))

                    for pred_num, (node, joint_prob) in enumerate(node_joint[:num_predictions], start=1):
                        # local_probability = softmax score within siblings
                        #                   = joint_prob / parent's joint_prob
                        parent_joint = get_joint_prob(node.parent, genome_probs)
                        local_prob = joint_prob / parent_joint if parent_joint > 0.0 else 0.0
                        rows.append({
                            "name": name,
                            "rank": rank,
                            "taxon": barbet.node_to_str(node),
                            "joint_probability": joint_prob,
                            "local_probability": local_prob,
                            "prediction_number": pred_num,
                            "in_predicted_lineage": node in greedy_path_set,
                        })

            else:
                # ── Non-global (conditional) mode ─────────────────────────────
                # Top-N alternatives are restricted to children of the greedy
                # top-1 parent at each rank.
                current_parent = tree
                # Track the cumulative probability of the greedy top-1 path so
                # that single-child nodes (absent from node_list_softmax) can
                # inherit it.
                current_prob = 1.0

                for rank in RANKS:
                    children = getattr(current_parent, "children", [])
                    if not children:
                        # Genuine leaf: taxonomy ends here.
                        break

                    valid_children = [c for c in children if c in node_to_col_idx]

                    if not valid_children:
                        # Single-child node: conditional probability is 1.0, so
                        # it inherits the running cumulative probability.
                        for pred_num, child in enumerate(children[:num_predictions], start=1):
                            rows.append({
                                "name": name,
                                "rank": rank,
                                "taxon": barbet.node_to_str(child),
                                "probability": current_prob,
                                "prediction_number": pred_num,
                            })
                        current_parent = children[0]
                        continue

                    child_probs = [
                        (child, float(genome_probs[node_to_col_idx[child]]))
                        for child in valid_children
                    ]
                    child_probs.sort(key=lambda x: (-x[1], barbet.node_to_str(x[0])))

                    for pred_num, (child, prob) in enumerate(child_probs[:num_predictions], start=1):
                        rows.append({
                            "name": name,
                            "rank": rank,
                            "taxon": barbet.node_to_str(child),
                            "probability": prob,
                            "prediction_number": pred_num,
                        })

                    current_prob = child_probs[0][1]
                    current_parent = child_probs[0][0]

        if global_predictions:
            return pl.DataFrame(
                rows,
                schema={
                    "name": pl.Utf8,
                    "rank": pl.Utf8,
                    "taxon": pl.Utf8,
                    "joint_probability": pl.Float64,
                    "local_probability": pl.Float64,
                    "prediction_number": pl.Int64,
                    "in_predicted_lineage": pl.Boolean,
                },
            )
        return pl.DataFrame(
            rows,
            schema={
                "name": pl.Utf8,
                "rank": pl.Utf8,
                "taxon": pl.Utf8,
                "probability": pl.Float64,
                "prediction_number": pl.Int64,
            },
        )




        




