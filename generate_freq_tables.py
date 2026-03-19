import pandas as pd
import spacy
from tqdm import tqdm
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_WITHOUT_ART = "/user/quzhengyi/dxy/output/Total_Freq_Without_Art.xlsx"
OUTPUT_WITH_ART = "/user/quzhengyi/dxy/output/Total_Freq_With_Art.xlsx"

os.makedirs(os.path.dirname(OUTPUT_WITH_ART), exist_ok=True)

# ==========================================
# 2. 核心配置：两套标签对比
# ==========================================
# 第一套：不包含 WORK_OF_ART
TARGET_LABELS_WITHOUT_ART = {"GPE", "ORG", "LAW", "EVENT", "NORP", "LOC", "MOVEMENT", "PRODUCT"}
# 第二套：包含 WORK_OF_ART
TARGET_LABELS_WITH_ART = {"GPE", "ORG", "LAW", "EVENT", "NORP", "LOC", "MOVEMENT", "PRODUCT", "WORK_OF_ART"}

ENTITY_ALIASES = {
    "国際連合": "国連", "安全保障理事会": "安保理", "国連安保理": "安保理",
    "国際連合憲章": "国連憲章", "ＰＫＯ": "PKO", "国連ＰＫＯ": "PKO", "国連平和維持活動": "PKO",
    "核兵器不拡散条約": "核不拡散条約(NPT)", "核不拡散条約": "核不拡散条約(NPT)", "ＮＰＴ": "核不拡散条約(NPT)",
    "核兵器実験禁止条約": "核実験禁止条約", "全面核実験禁止条約": "核実験禁止条約",
    "核兵器実験": "核実験", "核兵器戦争": "核戦争",
    "中近東": "中東", "アフガン": "アフガニスタン", "カンボジア": "カンボディア", 
    "ヴィエトナム": "ベトナム", "太平洋地域": "太平洋", "南アフリカ共和国": "南アフリカ",
    "日本国政府": "日本政府", "ＴＩＣＡＤ": "TICAD", "ＯＤＡ": "ODA", "ＵＨＣ": "UHC", 
    "ＭＤＧｓ": "MDGs", "ＳＤＧｓ": "SDGs", "米国": "アメリカ", "コロナ": "新型コロナ",
    "新型コロナウイルス": "新型コロナ", "中共": "中国", "ソ連": "ソビエト連邦", "露": "ロシア"
}

POLITICAL_STOP_WORDS = {
    "日本", "我が国", "世界", "国民", "政府", "国家", "国", "社会", "経済", "今日", 
    "今年", "皆様", "人々", "地域", "関係", "問題", "対応", "実現", "発展", "協力", 
    "推進", "取組", "取り組み", "改革", "政策", "分野", "状況", "必要", "日本国", 
    "全体", "国際社会", "時代", "会議", "平和", "皆様方"
}

# ==========================================
# 3. 加载模型与双轨数据统计
# ==========================================
print("正在配置 GPU 并加载 Transformer 模型...")
spacy.require_gpu()
nlp = spacy.load("ja_core_news_trf")
nlp.max_length = 5000000

df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])

# 用两个字典分别记录两套标准的词频
freq_without_art = {}
freq_with_art = {}

print("🚀 正在扫描文献以统计词频 (双轨计算)...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    text = str(row[1])
    if len(text) < 10: continue
    
    doc = nlp(text)
    
    for ent in doc.ents:
        label = ent.label_
        
        # 修正非洲的标签错位
        if ent.text.strip() == "アフリカ" and label == "GPE": 
            label = "LOC"
            
        entity_text = ent.text.strip()
        
        # 归一化
        if entity_text in ENTITY_ALIASES: 
            entity_text = ENTITY_ALIASES[entity_text]
            
        # 过滤单字和停用词
        if len(entity_text) > 1 and entity_text not in POLITICAL_STOP_WORDS:
            
            # 第一轨：如果属于旧的 8 个标签，记录到 Without 表
            if label in TARGET_LABELS_WITHOUT_ART:
                freq_without_art[entity_text] = freq_without_art.get(entity_text, 0) + 1
                
            # 第二轨：如果属于新的 9 个标签，记录到 With 表
            if label in TARGET_LABELS_WITH_ART:
                freq_with_art[entity_text] = freq_with_art.get(entity_text, 0) + 1

# ==========================================
# 4. 排序并输出为 Excel
# ==========================================
print("\n🧮 正在生成最终的实体排位表...")

# 处理 Without 表
df_without = pd.DataFrame(list(freq_without_art.items()), columns=["Entity", "Frequency"])
df_without = df_without.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
df_without.index = df_without.index + 1  # 让索引从 1 开始，代表 Rank 排名
df_without.index.name = "Rank"
df_without.to_excel(OUTPUT_WITHOUT_ART)

# 处理 With 表
df_with = pd.DataFrame(list(freq_with_art.items()), columns=["Entity", "Frequency"])
df_with = df_with.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
df_with.index = df_with.index + 1  # 让索引从 1 开始，代表 Rank 排名
df_with.index.name = "Rank"
df_with.to_excel(OUTPUT_WITH_ART)

print(f"✅ 【旧标准】不包含 WORK_OF_ART 的词频总榜已保存至:\n   -> {OUTPUT_WITHOUT_ART}")
print(f"✅ 【新标准】包含 WORK_OF_ART 的词频总榜已保存至:\n   -> {OUTPUT_WITH_ART}")