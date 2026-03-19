import pandas as pd
import spacy
from tqdm import tqdm
import os
from collections import defaultdict

# ==========================================
# 1. 核心配置与路径设定
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_EXCEL = "/user/quzhengyi/dxy/output/Phasic_Image_Only.xlsx"

os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)

TARGET_LABELS = {
    "GPE", "LOC", "ORG", "LAW", "EVENT", 
    "NORP", "MOVEMENT", "PRODUCT"
}

# 政治专属停用词表
POLITICAL_STOP_WORDS = {
    "日本", "我が国", "世界", "国民", "政府", "国家", "国",
    "社会", "経済", "今日", "今年", "皆様", "人々", "地域",
    "関係", "問題", "対応", "実現", "発展", "協力", "推進",
    "取組", "取り組み", "改革", "政策", "分野", "状況", "必要",
    "日本国", "全体", "国際社会", "時代", "会議", "平和", "皆様方"
}

# 实体归一化字典
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
    "ソ連": "ソビエト連邦", "露": "ロシア"
}
# ==========================================
# 2. 核心算法：带容错机制的最长连续跨度计算
# ==========================================
def get_max_continuous_span(years_list):
    """
    计算实体的最长连续出场时间（相邻年份间隔不超过 1 年视为连续）
    """
    if not years_list: 
        return 0
    
    sorted_years = sorted(list(set(years_list)))
    max_span, current_start, current_end = 0, sorted_years[0], sorted_years[0]
    
    for i in range(1, len(sorted_years)):
        current_year = sorted_years[i]
        
        # 核心逻辑：差值 <= 2，即容忍 1 年的断档
        if current_year - current_end <= 2: 
            current_end = current_year
        else:
            span_length = current_end - current_start + 1
            if span_length > max_span: 
                max_span = span_length
            current_start, current_end = current_year, current_year
            
    final_span = current_end - current_start + 1
    if final_span > max_span: 
        max_span = final_span
        
    return max_span

# ==========================================
# 3. 数据提取与统计
# ==========================================
print("正在加载 Spacy NLP 模型...")
try:
    spacy.require_gpu()
except:
    pass
nlp = spacy.load("ja_core_news_trf")

print("正在读取讲稿语料库...")
df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])

# 数据结构： entity -> { "total_freq": int, "years": set() }
entity_stats = defaultdict(lambda: {"total_freq": 0, "years": set()})

print("开始进行命名实体识别(NER)与词频统计...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    year = int(str(row[0]).strip())
    text = str(row[1])
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ in TARGET_LABELS:
            entity_text = ent.text.strip()
            if entity_text in ENTITY_ALIASES: 
                entity_text = ENTITY_ALIASES[entity_text]
                
            if len(entity_text) > 1 and entity_text not in POLITICAL_STOP_WORDS:
                entity_stats[entity_text]["total_freq"] += 1
                entity_stats[entity_text]["years"].add(year)

# ==========================================
# 4. 严格执行【阶段性形象】过滤逻辑
# ==========================================
print("开始应用最新阶段性形象标准过滤数据...")
phasic_results = []

for ent, stats in entity_stats.items():
    years_count = len(stats["years"])
    
    # 标准 ①：出现年数在 5 至 34 年之间
    if 5 <= years_count <= 34:
        
        # 计算带容错的最长连续跨度
        max_span = get_max_continuous_span(stats["years"])
        
        # 标准 ②：最长连续时段 >= 4 年
        if max_span >= 4:
            sorted_years_str = ", ".join(map(str, sorted(list(stats["years"]))))
            
            phasic_results.append({
                "实体 (Entity)": ent,
                "总频次 (Total Freq)": stats["total_freq"],
                "出场年数 (Years Count)": years_count,
                "最长连续跨度 (Max Span)": max_span,
                "具体出场年份 (Years List)": sorted_years_str
            })

df_phasic = pd.DataFrame(phasic_results)

# 按“出场年数”和“最长连续跨度”降序排列，让最具代表性的阶段议题排在前面
df_phasic = df_phasic.sort_values(by=["出场年数 (Years Count)", "最长连续跨度 (Max Span)"], ascending=[False, False])

# ==========================================
# 5. 导出结果
# ==========================================
df_phasic.to_excel(OUTPUT_EXCEL, index=False)

print(f"\n✅ 提取完成！共找到 {len(df_phasic)} 个符合【阶段性形象】标准的实体。")
print(f"📊 专属榜单已保存至: {OUTPUT_EXCEL}")