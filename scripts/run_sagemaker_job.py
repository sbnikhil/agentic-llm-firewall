import os
from sagemaker.huggingface import HuggingFace
from dotenv import load_dotenv
load_dotenv()

BUCKET          = os.getenv("S3_BUCKET", "llm-firewall")
ROLE_ARN        = os.environ["SAGEMAKER_ROLE_ARN"]   # set in .env, never in source
REGION          = os.getenv("AWS_REGION", "us-east-1")
INSTANCE_TYPE   = "ml.g4dn.xlarge"   # T4 GPU — cheapest GPU option


def main():
    print("Submitting SageMaker fine-tuning job...")

    # Data locations in S3
    train_s3 = f"s3://{BUCKET}/finetune"
    val_s3   = f"s3://{BUCKET}/finetune"

    estimator = HuggingFace(
        entry_point="finetune_train.py",
        source_dir="training",   # only the training script and its container requirements are uploaded
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        output_path=f"s3://{BUCKET}/finetune-output",
        base_job_name="llm-firewall-deberta",
        hyperparameters={
            "model_name":    "ProtectAI/deberta-v3-base-prompt-injection-v2",
            "epochs":        3,
            "batch_size":    16,
            "learning_rate": 2e-5,
            "max_length":    256,
        },
        environment={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT", "llm-firewall-injection-detection"),
        },
        region_name=REGION,
    )

    estimator.fit(
        inputs={
            "train": train_s3,
            "val":   val_s3,
        },
        wait=False,   # returns immediately — monitor in SageMaker console
    )

    print(f"\nJob submitted: {estimator.latest_training_job.name}")
    print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")
    print(f"Output will be at: s3://{BUCKET}/finetune-output/")
    print("\nRun this to check status:")
    print(f"  aws sagemaker describe-training-job --training-job-name {estimator.latest_training_job.name} --query 'TrainingJobStatus'")


if __name__ == "__main__":
    main()
