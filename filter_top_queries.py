import pandas as pd
from pathlib import Path
from collections import Counter
import re

def extract_ngrams(text, n):
    """Bir metinden n-gram'ları çıkar."""
    words = str(text).lower().split()
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def filter_top_queries():
    # Dosya yolları
    input_path = Path("output/related_queries.csv")
    output_path = Path("output/filtered_top_queries.csv")

    if not input_path.exists():
        print(f"Hata: {input_path} dosyası bulunamadı!")
        return

    print(f"'{input_path}' okunuyor (sadece 'top' türü)...")
    
    # CSV dosyasını oku
    df = pd.read_csv(input_path)

    # 1. Sadece 'type' sütunu 'top' olanları filtrele
    top_df = df[df["type"] == "top"].copy()

    # 2. Sadece Latin alfabesi karakterleri, sayılar, boşluklar ve yaygın noktalama işaretleri
    # Latin Extended (Örn: é, ö, ş) dahil
    latin_pattern = re.compile(r'^[A-Za-z0-9\s\.,!\?\-\(\)\'\"&/@#\$%]+$')
    
    def is_latin_alphabet(text):
        if not isinstance(text, str) or text.strip() == "":
            return False
        return bool(latin_pattern.match(text))

    # 3. Value sütununu sayısal değere çevir ve 10'un altındakileri filtrele
    top_df["value"] = pd.to_numeric(top_df["value"], errors='coerce').fillna(0)
    top_df = top_df[top_df["value"] >= 10]

    # 4. Latin alfabesi kontrolü
    print("Alfabe kontrolü yapılıyor...")
    from tqdm import tqdm
    tqdm.pandas()
    top_df = top_df[top_df["query"].progress_apply(is_latin_alphabet)]

    # 5. Aynı sorgu (exact match) olanlarda, kategoriye bakmaksızın en yüksek value'su olanı tut
    print("Tekilleştirme yapılıyor (Kategori bağımsız)...")
    # Önce value'ya göre büyükten küçüğe sıralıyoruz
    deduplicated_df = top_df.sort_values(by=["query", "value"], ascending=[True, False]) \
                             .drop_duplicates(subset=["query"], keep="first")

    # ---------------------------------------------------------------
    # 6. GRUPLAMA MANTIĞI (N-gram Analizi)
    # ---------------------------------------------------------------
    print("Gruplama için n-gram analizi yapılıyor...")
    queries = deduplicated_df["query"].dropna().str.strip().str.lower().tolist()
    
    trigram_counter = Counter()
    bigram_counter = Counter()
    unigram_counter = Counter()

    for q in queries:
        trigram_counter.update(set(extract_ngrams(q, 3)))
        bigram_counter.update(set(extract_ngrams(q, 2)))
        unigram_counter.update(set(extract_ngrams(q, 1)))

    MIN_FREQ = 3
    common_trigrams = {phrase for phrase, count in trigram_counter.items() if count >= MIN_FREQ}
    common_bigrams  = {phrase for phrase, count in bigram_counter.items()  if count >= MIN_FREQ}
    common_unigrams = {phrase for phrase, count in unigram_counter.items() if count >= MIN_FREQ}
    
    STOP_WORDS = {
        "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "it",
        "on", "at", "by", "with", "from", "as", "are", "was", "be", "has",
        "had", "have", "do", "does", "did", "not", "no", "but", "if", "can",
        "will", "how", "what", "when", "where", "who", "which", "why",
        "this", "that", "these", "those", "my", "your", "his", "her", "its",
        "our", "their", "i", "you", "he", "she", "we", "they", "me", "us",
        "vs", "top", "best", "new", "most", "about", "all", "more",
    }
    common_unigrams -= STOP_WORDS

    def assign_group(query):
        q = str(query).lower().strip()
        matches_3 = [ng for ng in extract_ngrams(q, 3) if ng in common_trigrams]
        if matches_3:
            return max(matches_3, key=lambda x: trigram_counter[x])
        matches_2 = [ng for ng in extract_ngrams(q, 2) if ng in common_bigrams]
        if matches_2:
            return max(matches_2, key=lambda x: bigram_counter[x])
        matches_1 = [ng for ng in extract_ngrams(q, 1) if ng in common_unigrams]
        if matches_1:
            return max(matches_1, key=lambda x: unigram_counter[x])
        return q

    print("Gruplar atanıyor...")
    deduplicated_df["group"] = deduplicated_df["query"].progress_apply(assign_group)

    # 6.5. GRUP FİLTRESİ: Sadece 1 adet satırı olan grupları sil
    print("Benzersiz (tek kalan) gruplar temizleniyor...")
    group_counts = deduplicated_df["group"].value_counts()
    groups_to_keep = group_counts[group_counts > 1].index
    deduplicated_df = deduplicated_df[deduplicated_df["group"].isin(groups_to_keep)]

    # 6.6. 'Real Estate' içeren grupları temizle
    print("'Real Estate' içeren gruplar siliniyor...")
    deduplicated_df = deduplicated_df[~deduplicated_df["group"].str.contains("real estate", case=False, na=False)]

    # ---------------------------------------------------------------
    # 7. SIRALAMA VE KAYDETME
    # ---------------------------------------------------------------
    output_columns = ["group", "category_name", "type", "query", "value"]
    # Önce gruba (A-Z), sonra o grup içindeki value değerine (Z-A) göre sırala
    final_df = deduplicated_df[output_columns].sort_values(by=["group", "value", "query"], ascending=[True, False, True])

    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 8. BENZERSİZ GRUPLARI .TXT OLARAK KAYDET
    # ---------------------------------------------------------------
    unique_groups = sorted(final_df["group"].unique())
    txt_path = Path("output/unique_groups.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join([f'"{g}"' for g in unique_groups]))
    
    print(f"\nİşlem tamamlandı!")
    print(f"Başlangıç satır sayısı (top): {len(top_df)}")
    print(f"Filtreleme & Tekilleştirme sonrası: {len(deduplicated_df)}")
    print(f"Benzersiz grup sayısı: {len(unique_groups)}")
    print(f"Sonuç dosyası: {output_path}")
    print(f"Grup listesi: {txt_path}")

if __name__ == "__main__":
    filter_top_queries()
