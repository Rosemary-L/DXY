import pandas as pd
import spacy
from tqdm import tqdm
from collections import Counter

# ==========================================
# 1. 基础配置与路径
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"   
OUTPUT_CSV = "/user/quzhengyi/dxy/output/Global_Top50_Normalized.csv" 
OUTPUT_EXCEL = "/user/quzhengyi/dxy/output/Global_Top50_Normalized.xlsx" 

TARGET_LABELS = {"GPE", "ORG", "LAW", "EVENT", "NORP", "LOC", "MOVEMENT", "PRODUCT"}

# 【新增核心配置】：实体归一化映射字典 (Alias Mapping)
# 格式为 {"别名/简称": "标准统一名称"}
# 你可以在跑出初步结果后，把发现的重合词不断补充到这个字典里
ENTITY_ALIASES = {
    "安保理": "安全保障理事会",
    "米国": "アメリカ",
    "ＵＮ": "国連",
    "国際連合": "国連",
    "コロナ": "新型コロナ",
    "新型コロナウイルス": "新型コロナ",
    "中共": "中国",
    "ソ連": "ソビエト連邦",
    "露": "ロシア",
    # 在这里继续添加你发现的其他同义词...
}

# ==========================================
# 2. 加载模型与 GPU 配置
# ==========================================
print("正在配置 GPU 并加载 Transformer 模型...")
try:
    spacy.require_gpu()
except Exception as e:
    pass
nlp = spacy.load("ja_core_news_trf")

# ==========================================
# 3. 读取数据与初始化
# ==========================================
df = pd.read_csv(INPUT_CSV, header=None, names=["Year", "Text"])
global_entity_counter = Counter()

# ==========================================
# 4. 执行全局实体抽取与【归一化】
# ==========================================
print(f"开始处理 {len(df)} 篇讲稿数据...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row["Text"])
    if pd.isna(text) or len(text.strip()) == 0: continue
        
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ in TARGET_LABELS and len(ent.text) > 1:
            raw_entity_text = ent.text.strip()
            
            # 【核心逻辑】：实体归一化
            # .get(A, B) 的意思是：如果在字典里找到了 A，就替换成标准名；如果没找到，就保持原样(B)
            normalized_entity = ENTITY_ALIASES.get(raw_entity_text, raw_entity_text)
            
            # 将归一化后的实体存入 Counter
            global_entity_counter[(normalized_entity, ent.label_)] += 1

# ==========================================
# 5. 提取 Top 50 并保存
# ==========================================
top_50_entities = global_entity_counter.most_common(50)

final_rows = []
for (entity, label), count in top_50_entities:
    final_rows.append({
        "Rank": len(final_rows) + 1,
        "Normalized_Entity": entity, # 这里的实体已经是合并后的标准名了
        "Label": label,
        "Frequency": count
    })

result_df = pd.DataFrame(final_rows)

print("正在导出文件...")
result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
result_df.to_excel(OUTPUT_EXCEL, index=False)
print("归一化提取完成！")