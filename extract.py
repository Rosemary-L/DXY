import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_CSV = "/user/quzhengyi/dxy/output/overall_top15.csv"

# 确保输出的文件夹存在
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ==========================================
# 2. 核心配置：目标标签与完整归一化映射字典
# ==========================================
TARGET_LABELS = {"GPE", "ORG", "LAW", "EVENT", "NORP", "LOC", "MOVEMENT", "PRODUCT"}

# 根据用户提供的 25 组规则生成的强归一化字典
ENTITY_ALIASES = {
    # 联合国机构系列
    "国際連合": "国連",
    "安全保障理事会": "安保理",
    "国連安保理": "安保理",
    "国際連合憲章": "国連憲章",
    "国連憲章再検討問題": "国連憲章",
    "国連憲章調印": "国連憲章",
    "国連憲章の原則": "国連憲章",
    "ＰＫＯ": "PKO",
    "国連ＰＫＯ": "PKO",
    "国連平和維持活動": "PKO",

    # 核议题系列
    "核兵器不拡散条約": "核不拡散条約(NPT)",
    "核不拡散条約": "核不拡散条約(NPT)",
    "核兵器拡散防止条約": "核不拡散条約(NPT)",
    "核拡散防止条約": "核不拡散条約(NPT)",
    "ＮＰＴ": "核不拡散条約(NPT)",
    "核兵器不拡散条約体制": "核不拡散体制",
    "核不拡散条約体制": "核不拡散体制",
    "核不拡散": "核不拡散体制",
    "核軍縮 ・ 不拡散体制": "核不拡散体制",
    "ＮＰＴ体制": "核不拡散体制",
    "核兵器実験禁止条約": "核実験禁止条約",
    "全面核実験禁止条約": "核実験禁止条約",
    "全面核実験禁止条約交渉": "核実験禁止条約",
    "核実験停止条約": "核実験禁止条約",
    "核兵器実験": "核実験",
    "核実験全面禁止": "核実験",
    "核実験検証制度": "核実験",
    "核実験２条約": "核実験",
    "部分核停条約": "核実験",
    "核実験中止協定": "核実験中止",
    "核実験停止協定": "核実験中止",
    "核兵器実験停止協定": "核実験中止",
    "ジュネーヴ核実験中止会議": "核実験中止",
    "ジュネーブ核実験中止会議": "核実験中止",
    "核実験中止問題": "核実験中止",
    "核実験中止会議": "核実験中止",
    "核兵器戦争": "核戦争",
    "中距離核戦力交渉": "核戦争",

    # 地名系列
    "中近東": "中東",
    "中東地域": "中東",
    "アフガン": "アフガニスタン",
    "カンボジア": "カンボディア",
    "ヴィエトナム": "ベトナム",
    "ヴィエトナム社会主義共和国": "ベトナム",
    "太平洋地域": "太平洋",
    "南アフリカ共和国": "南アフリカ",

    # 机构/政策系列
    "日本国政府": "日本政府",
    "日本国代表団": "日本代表団",
    "ＴＩＣＡＤ": "TICAD",
    "TICADIV": "TICAD",
    "ＯＤＡ": "ODA",
    "ＵＨＣ": "UHC",
    "ユニバーサル ・ ヘルス ・ カバレッジ": "UHC",
    "ＭＤＧｓ": "MDGs",
    "ＳＤＧｓ": "SDGs",
    "UNウィメン": "UN Women",
    
    # 之前遗留的必要补充（为了防止基础词汇分散）
    "米国": "アメリカ",
    "コロナ": "新型コロナ",
    "新型コロナウイルス": "新型コロナ",
    "中共": "中国",
    "ソ連": "ソビエト連邦",
    "露": "ロシア"
}

# ==========================================
# 3. 加载 GPU 与模型
# ==========================================
print("正在配置 GPU 并加载 Transformer 模型...")
spacy.require_gpu()
nlp = spacy.load("ja_core_news_trf")
nlp.max_length = 5000000

# ==========================================
# 4. 执行核心提取与统计逻辑
# ==========================================
print("正在读取原始讲稿数据...")
df_raw = pd.read_csv(INPUT_CSV, header=None)
all_extracted_data = []

print("🚀 开始逐年提取实体，正在执行 25 组归一化规则...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    year = str(row[0]).strip()
    text = str(row[1])
    
    if pd.isna(text) or len(text) < 10:
        continue
        
    doc = nlp(text)
    year_entities = []
    
    for ent in doc.ents:
        label = ent.label_
        
        # 针对文档特殊要求的标签统一处理（非洲 GPE 转 LOC）
        if ent.text.strip() == "アフリカ" and label == "GPE":
            label = "LOC"
            
        if label in TARGET_LABELS:
            entity_text = ent.text.strip()
            
            # 查字典，执行归一化
            if entity_text in ENTITY_ALIASES:
                entity_text = ENTITY_ALIASES[entity_text]
                
            # 噪音过滤：去掉长度为1的无意义单字
            if len(entity_text) > 1:
                year_entities.append((entity_text, label))
                
    counter = Counter(year_entities)
    top_15 = counter.most_common(15)
    
    for (ent_text, ent_label), freq in top_15:
        all_extracted_data.append({
            "Year": year,
            "Entity": ent_text,
            "Label": ent_label,
            "Frequency": freq
        })

# ==========================================
# 5. 保存最终结果
# ==========================================
df_result = pd.DataFrame(all_extracted_data)
df_result.to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 大功告成！历年 Top 15 实体已提取完毕，并保存至:\n{OUTPUT_CSV}")