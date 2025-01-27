from train_reft import TrainingConfig, main
import os

def test_training():
    # 确保输出目录存在
    os.makedirs("./output", exist_ok=True)
    
    # 创建配置
    config = TrainingConfig(
        model_name_or_path="meta-llama/Llama-2-7b-chat-hf",  # 或其他支持chat template的模型
        tokenizer_name_or_path="meta-llama/Llama-2-7b-chat-hf",
        model_dir="./output",
        train_file="train_data.json",
        test_file="test_data.json",
        batch_size=4,
        eval_batch_size=4,
        num_workers=2,
        num_epochs=2,  # 测试时用较少的epoch
        max_input_length=512,
        max_gen_length=512,
        wandb_log=False  # 测试时关闭wandb
    )
    
    main(config)

if __name__ == "__main__":
    test_training() 