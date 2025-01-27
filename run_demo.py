import json
from train_reft import TrainingArgs, main

def generate_demo_data():
    """
    生成最小可行的训练集、测试集，用来演示意图识别强化学习流程。
    """
    train_data = [
        {"question": "今天天气怎么样？", "label": 1},
        {"question": "请播放一段音乐", "label": 2},
    ]
    test_data = [
        {"question": "如何联系人工客服？", "label": 3},
    ]
    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

def run_demo():
    args = TrainingArgs(
        model_name_or_path="gemma-2-2b-it",      # 需替换为你本地可用的模型
        ref_model_name_or_path="gemma-2-2b-it",  # 同上
        tokenizer_name_or_path="gemma-2-2b-it",  # 同上
        output_dir="./ppo_peft_output_demo",
        train_file="train.json",
        test_file="test.json",
        max_input_length=64,
        max_gen_length=32,
        train_batch_size=1,
        eval_batch_size=1,
        n_epochs=1,
        kl_coef=0.1,
        keep_num_ckpt=1,
    )
    main(args)

if __name__ == "__main__":
    generate_demo_data()
    run_demo()
    print("演示完毕！") 