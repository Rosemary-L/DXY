import pandas as pd
import spacy
from tqdm import tqdm
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_EXCEL = "/user/quzhengyi/dxy/output/Image4_Sudden_Events.xlsx"

os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)

# ==========================================
# 2. 核心配置：归一化字典与停用词清单
# ==========================================
TARGET_LABELS = {"GPE", "ORG", "LAW", "EVENT", "NORP", "LOC", "MOVEMENT", "PRODUCT"}

ENTITY_ALIASES = {
    "国際連合": "国連", "安全保障理事会": "安保理", "国連安保理": "安保理",
    "国際連合憲章": "国連憲章", "国連憲章再検討問題": "国連憲章", "国連憲章調印": "国連憲章", "国連憲章の原則": "国連憲章",
    "ＰＫＯ": "PKO", "国連ＰＫＯ": "PKO", "国連平和維持活動": "PKO",
    "核兵器不拡散条約": "核不拡散条約(NPT)", "核不拡散条約": "核不拡散条約(NPT)",
    "核兵器拡散防止条約": "核不拡散条約(NPT)", "核拡散防止条約": "核不拡散条約(NPT)", "ＮＰＴ": "核不拡散条約(NPT)",
    "核兵器不拡散条約体制": "核不拡散体制", "核不拡散条約体制": "核不拡散体制", "核不拡散": "核不拡散体制",
    "核軍縮 ・ 不拡散体制": "核不拡散体制", "ＮＰＴ体制": "核不拡散体制",
    "核兵器実験禁止条約": "核実験禁止条約", "全面核実験禁止条約": "核実験禁止条約",
    "全面核実験禁止条約交渉": "核実験禁止条約", "核実験停止条約": "核実験禁止条約",
    "核兵器実験": "核実験", "核実験全面禁止": "核実験", "核実験検証制度": "核実験",
    "核実験２条約": "核実験", "部分核停条約": "核実験",
    "核実験中止協定": "核実験中止", "核実験停止協定": "核実験中止", "核兵器実験停止協定": "核実験中止",
    "ジュネーヴ核実験中止会議": "核実験中止", "ジュネーブ核実験中止会議": "核実験中止",
    "核実験中止問題": "核実験中止", "核実験中止会議": "核実験中止",
    "核兵器戦争": "核戦争", "中距離核戦力交渉": "核戦争",
    "中近東": "中東", "中東地域": "中東", "アフガン": "アフガニスタン",
    "カンボジア": "カンボディア", "ヴィエトナム": "ベトナム", "ヴィエトナム社会主義共和国": "ベトナム",
    "太平洋地域": "太平洋", "南アフリカ共和国": "南アフリカ",
    "日本国政府": "日本政府", "日本国代表団": "日本代表団",
    "ＴＩＣＡＤ": "TICAD", "TICADIV": "TICAD", "ＯＤＡ": "ODA",
    "ＵＨＣ": "UHC", "ユニバーサル ・ ヘルス ・ カバレッジ": "UHC",
    "ＭＤＧｓ": "MDGs", "ＳＤＧｓ": "SDGs", "UNウィメン": "UN Women",
    "米国": "アメリカ", "コロナ": "新型コロナ", "新型コロナウイルス": "新型コロナ",
    "中共": "中国", "ソ連": "ソビエト連邦", "露": "ロシア"
}

POLITICAL_STOP_WORDS = {
    "日本", "我が国", "世界", "国民", "政府", "国家", "国", 
    "社会", "経済", "今日", "今年", "皆様", "人々", "地域", 
    "関係", "問題", "対応", "実現", "発展", "協力", "推進", 
    "取組", "取り組み", "改革", "政策", "分野", "状況", "必要", 
    "日本国", "全体", "国際社会", "時代", "会議", "平和", "皆様方"
}

# ==========================================
# 3. 加载 GPU 与模型
# ==========================================
print("正在配置 GPU 并加载 Transformer 模型...")
spacy.require_gpu()
nlp = spacy.load("ja_core_news_trf")
nlp.max_length = 5000000

# ==========================================
# 4. 执行全局提取，保留完整的年份轨迹
# ==========================================
print("📥 正在读取原始讲稿数据...")
df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])

entity_yearly_freq = {}
years_in_doc = set()

print("🚀 开始扫描历史文献，追踪时间序列...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    year = int(str(row[0]).strip())
    text = str(row[1])
    years_in_doc.add(year)
    
    if len(text) < 10: continue
        
    doc = nlp(text)
    
    for ent in doc.ents:
        label = ent.label_
        if ent.text.strip() == "アフリカ" and label == "GPE":
            label = "LOC"
            
        if label in TARGET_LABELS:
            entity_text = ent.text.strip()
            if entity_text in ENTITY_ALIASES:
                entity_text = ENTITY_ALIASES[entity_text]
                
            if len(entity_text) > 1 and entity_text not in POLITICAL_STOP_WORDS:
                if entity_text not in entity_yearly_freq:
                    entity_yearly_freq[entity_text] = {"Label": label, "Yearly_Data": {}}
                
                current_count = entity_yearly_freq[entity_text]["Yearly_Data"].get(year, 0)
                entity_yearly_freq[entity_text]["Yearly_Data"][year] = current_count + 1

# ==========================================
# 5. 执行【突发性事件】探测算法
# ==========================================
print("\n🧮 正在执行【突发事件/黑天鹅】雷达探测 (当年≥10次 且 达到前3年均值3倍)...")

min_yr = min(years_in_doc)
max_yr = max(years_in_doc)

sudden_events = []

for ent, data in entity_yearly_freq.items():
    yearly_data = data["Yearly_Data"]
    label = data["Label"]
    
    # 从数据的第 4 年开始扫描（因为要看前 3 年均值）
    for current_year in range(min_yr + 3, max_yr + 1):
        freq_current = yearly_data.get(current_year, 0)
        
        # 指标 1：绝对底线，当年必须爆发 >= 10 次
        if freq_current >= 10:
            # 获取前 3 年的频次
            freq_y1 = yearly_data.get(current_year - 1, 0)
            freq_y2 = yearly_data.get(current_year - 2, 0)
            freq_y3 = yearly_data.get(current_year - 3, 0)
            
            # 计算前 3 年均值
            prev_3_years_avg = (freq_y1 + freq_y2 + freq_y3) / 3.0
            
            # 指标 2：相对倍数判定
            is_sudden = False
            if prev_3_years_avg == 0:
                is_sudden = True # 平地惊雷 (前三年为0，今年猛增到10以上)
            elif freq_current >= (prev_3_years_avg * 3):
                is_sudden = True # 旱地拔葱 (达到均值的3倍以上)
                
            if is_sudden:
                sudden_events.append({
                    "Entity": ent,
                    "Label": label,
                    "Spike_Year": current_year, # 爆发年份
                    "Freq_In_Spike_Year": freq_current, # 当年爆发次数
                    "Prev_3_Years_Avg": round(prev_3_years_avg, 1), # 前3年均值
                    # 记录一下前三年的具体数据，方便你写论文时核对
                    "History_Detail": f"[{current_year-3}]:{freq_y3}, [{current_year-2}]:{freq_y2}, [{current_year-1}]:{freq_y1}"
                })

# 转化为 DataFrame，优先按“爆发年份”排序，方便你按历史时间线查看
df_sudden = pd.DataFrame(sudden_events)

if not df_sudden.empty:
    df_sudden = df_sudden.sort_values(by=["Spike_Year", "Freq_In_Spike_Year"], ascending=[True, False])
    df_sudden.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n🎉 探测完成！共捕捉到 {len(df_sudden)} 次【历史级突发事件】！")
    print(f"✅ 结果已保存至: {OUTPUT_EXCEL}")
else:
    print("\n⚠️ 哎呀，没有探测到符合条件的突发事件，可能是 10 次的底线对于你的语料有点高，可以尝试调成 5 次。")