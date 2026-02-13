import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_descriptions(file_path):
    model_a, model_b = None, None
    with open(file_path, encoding="utf-8") as f:
        for results in [json.loads(line) for line in f if line.strip()]:
            if "error" in results["result"] and "raw" in results["result"]:
                continue
            for res in results["result"]:
                if res["model"] == "Model A" or res["model"] == "A":
                    model_a = res["model_name"]
                elif res["model"] == "Model B" or res["model"] == "B":
                    model_b = res["model_name"]  #
            if model_a is not None and model_b is not None:
                break
    assert model_a is not None and model_b is not None, "Could not determine model names."

    descriptions = {("Model A", model_a): [], ("Model B", model_b): []}
    hash_index = []
    meta_info = []
    with open(file_path, encoding="utf-8") as f:
        for results in [json.loads(line) for line in f if line.strip()]:
            if "error" in results["result"] and "raw" in results["result"]:
                continue
            hash = results["hash"]
            for res_idx, res in enumerate(results["result"]):
                model_a_desc = model_b_desc = False
                if ("Model A" in res["property_description"] and "Model B" not in res["property_description"]) or (
                    res["property_description"].find("Model A") >= 0
                    and res["property_description"].find("Model A") < res["property_description"].find("Model B")
                ):
                    descriptions[("Model A", model_a)].append(res["property_description"])
                    model_a_desc = True
                elif ("Model B" in res["property_description"] and "Model A" not in res["property_description"]) or (
                    res["property_description"].find("Model B") >= 0
                    and res["property_description"].find("Model B") < res["property_description"].find("Model A")
                ):
                    descriptions[("Model B", model_b)].append(res["property_description"])
                    model_b_desc = True
                else:
                    raise ValueError(
                        f"Could not determine which model the description belongs to: {res['property_description']}"
                    )

                if "not_property_description" in res:
                    if (
                        "Model A" not in res["not_property_description"]
                        and "Model B" not in res["not_property_description"]
                    ):
                        if res["not_property_description"][:2] in ["A ", "A'"] or res["not_property_description"][
                            :2
                        ] in ["B ", "B'"]:
                            res["not_property_description"] = "Model " + res["not_property_description"]
                        else:
                            raise ValueError(
                                f"Could not determine which model the description belongs to: {res['not_property_description']}"
                            )

                    if (
                        "Model A" in res["not_property_description"]
                        and "Model B" not in res["not_property_description"]
                    ) or (
                        res["not_property_description"].find("Model A") >= 0
                        and res["not_property_description"].find("Model A")
                        < res["not_property_description"].find("Model B")
                    ):
                        descriptions[("Model A", model_a)].append(res["not_property_description"])
                        model_a_desc = True
                    elif (
                        "Model B" in res["not_property_description"]
                        and "Model A" not in res["not_property_description"]
                    ) or (
                        res["not_property_description"].find("Model B") >= 0
                        and res["not_property_description"].find("Model B")
                        < res["not_property_description"].find("Model A")
                    ):
                        descriptions[("Model B", model_b)].append(res["not_property_description"])
                        model_b_desc = True
                    else:
                        raise ValueError(
                            f"Could not determine which model the description belongs to: {res['not_property_description']}"
                        )

                if not (model_a_desc and model_b_desc):  # invalid contrastive pair
                    if model_a_desc:
                        descriptions[("Model A", model_a)] = descriptions[("Model A", model_a)][
                            : len(descriptions[("Model B", model_b)])
                        ]
                    elif model_b_desc:
                        descriptions[("Model B", model_b)] = descriptions[("Model B", model_b)][
                            : len(descriptions[("Model A", model_a)])
                        ]
                    assert len(descriptions[("Model A", model_a)]) == len(descriptions[("Model B", model_b)]), (
                        "Mismatched number of descriptions after adjustment."
                    )
                if model_a_desc and model_b_desc:
                    hash_index.append((hash, res_idx))
                    meta_info.append(
                        {
                            "category": res.get("category", ""),
                            "type": res.get("type", ""),
                            "impact": res.get("impact", ""),
                            "user_preference_direction": res.get("user_preference_direction", ""),
                            "contains_errors": res.get("contains_errors", ""),
                            "unexpected_behavior": res.get("unexpected_behavior", ""),
                        }
                    )

    assert len(descriptions[("Model A", model_a)]) == len(descriptions[("Model B", model_b)]), (
        "Mismatched number of descriptions between models."
    )
    assert len(hash_index) == len(descriptions[("Model A", model_a)]), (
        "Hash index length does not match number of descriptions."
    )
    assert len(meta_info) == len(descriptions[("Model A", model_a)]), (
        "Meta info length does not match number of descriptions."
    )

    return descriptions, hash_index, meta_info


def embed_sentences(sent_list, tokenizer, model, batch_size=64, device="cpu"):
    """
    sent_list: List[str]
    returns: torch.Tensor [N, dim] (L2-normalized)
    """
    all_embeddings = []

    for i in tqdm(
        range(0, len(sent_list), batch_size), desc="Embedding", total=(len(sent_list) + batch_size - 1) // batch_size
    ):
        batch_sents = sent_list[i : min(i + batch_size, len(sent_list))]
        encoded_input = tokenizer(batch_sents, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        all_embeddings.append(sentence_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_sentence_embeddings(
    difference_results: str,
    sentence_embedder: str,
    batch_size: int = 64,
    device: str = "cpu",
    regenerate: bool = False,
):
    assert os.path.isfile(difference_results)
    descriptions, hash_index, _meta_info = load_descriptions(difference_results)
    assert len(descriptions) > 0, "No valid descriptions found."
    assert len(descriptions) == 2, "Expected descriptions from exactly two models."

    tokenizer = AutoTokenizer.from_pretrained(sentence_embedder)
    embed_model = AutoModel.from_pretrained(sentence_embedder, device_map=device, trust_remote_code=True).eval()
    embed_model = embed_model.to(dtype=torch.bfloat16)

    output_dir = os.path.dirname(difference_results)
    os.makedirs(output_dir, exist_ok=True)

    out_paths = []
    for _idx, (key, texts) in enumerate(descriptions.items()):
        model_id, model_name = key

        safe_model_name = "".join(c if c.isalnum() or c in "-._" else "_" for c in model_name)
        out_path = os.path.join(output_dir, f"{model_id.replace(' ', '')}__{safe_model_name}.npz")
        out_paths.append(out_path)

        if os.path.isfile(out_path) and not regenerate:
            continue
        else:
            # replace model_id with placeholders
            texts_with_placeholder = [text.replace(model_id, "<MODEL>") for text in texts]
            other_model_id = "Model B" if model_id == "Model A" else "Model A"
            texts_with_placeholder = [text.replace(other_model_id, "<OTHER MODEL>") for text in texts_with_placeholder]

            embeddings = embed_sentences(
                texts_with_placeholder, tokenizer, embed_model, batch_size=batch_size, device=device
            )

            assert len(texts) == len(hash_index) == embeddings.size(0), (
                "Mismatch in lengths of texts, hash index, and embeddings."
            )
            np.savez_compressed(
                out_path,
                sentences=np.array(texts, dtype=object),
                sentences_with_placeholders=np.array(texts_with_placeholder, dtype=object),
                hash_index=np.array(hash_index, dtype=object),
                model_name=np.array([(model_id, model_name)] * len(texts), dtype=object),
                embeddings=embeddings.numpy(),
            )

    del embed_model
    del tokenizer
    torch.cuda.empty_cache()

    return out_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute sentence embeddings.")
    parser.add_argument("--sentence_embedder", type=str, default="intfloat/e5-large-v2", help="Pre-trained model name.")
    parser.add_argument(
        "--difference_results", type=str, required=True, help="Path to the JSONL file containing difference results."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on."
    )
    parser.add_argument(
        "--regenerate", action="store_true", help="Whether to regenerate embeddings even if they exist."
    )
    args = parser.parse_args()

    compute_sentence_embeddings(
        difference_results=args.difference_results,
        sentence_embedder=args.sentence_embedder,
        batch_size=args.batch_size,
        device=args.device,
        regenerate=args.regenerate,
    )
