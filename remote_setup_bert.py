import os

from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "bert-base-german-cased"
MODEL_DIR = os.path.join(
    "/var/lib/docker/wd/Sign/codes/VAC_CSLR",
    "pretrained",
    MODEL_NAME,
)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print("saved tokenizer and model to {}".format(MODEL_DIR))
    print("hidden_size={}".format(model.config.hidden_size))


if __name__ == "__main__":
    main()
