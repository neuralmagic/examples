from deepsparse.sentence_transformers import DeepSparseSentenceTransformer
from mteb import MTEB
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--task-names", type=str, nargs='+', required=True)
parser.add_argument("--output-dir", type=str, required=True)

def run_task(model_name, task_names, output_dir):
    model = DeepSparseSentenceTransformer(model_name)
    evaluation = MTEB(tasks=task_names)
    results = evaluation.run(model, output_folder=output_dir, batch_size=1)
    print(results)

if __name__ == "__main__":
    args = parser.parse_args()
    run_task(args.model_name, args.task_names, args.output_dir)