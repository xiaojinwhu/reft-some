import json
import random

def generate_math_problem():
    """生成简单的数学问题及其解答"""
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    operation = random.choice(['+', '-', '*'])
    
    question = f"计算 {num1} {operation} {num2} 的结果是多少？"
    
    if operation == '+':
        result = num1 + num2
        cot = f"让我一步步思考：\n1) 这是一个加法问题\n2) {num1} + {num2} = {result}\n因此，答案是 {result}。"
    elif operation == '-':
        result = num1 - num2
        cot = f"让我一步步思考：\n1) 这是一个减法问题\n2) {num1} - {num2} = {result}\n因此，答案是 {result}。"
    else:
        result = num1 * num2
        cot = f"让我一步步思考：\n1) 这是一个乘法问题\n2) {num1} × {num2} = {result}\n因此，答案是 {result}。"
    
    return {
        "question": question,
        "answer_cot": cot,
        "answer_value": float(result),
        "item_id": f"math_{random.randint(1000, 9999)}"
    }

def generate_dataset(num_samples: int, output_file: str):
    """生成数据集并保存到文件"""
    dataset = [generate_math_problem() for _ in range(num_samples)]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 生成训练集和测试集
    generate_dataset(100, "train_data.json")
    generate_dataset(20, "test_data.json")
    print("数据生成完成！") 