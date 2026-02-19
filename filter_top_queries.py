import pandas as pd
from pathlib import Path

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

    # Sadece 'type' sütunu 'top' olanları filtrele
    top_df = df[df["type"] == "top"]

    # İstenen sütunları seç
    selected_columns = ["category_name", "type", "query", "value"]
    
    # Sütunların varlığını kontrol et
    missing_cols = [col for col in selected_columns if col not in top_df.columns]
    if missing_cols:
        print(f"Hata: Dosyada şu sütunlar eksik: {missing_cols}")
        return

    # Aynı kategori ve aynı sorgu (exact match) olanlarda value'su en yüksek olanı tut
    filtered_df = top_df[selected_columns].copy()
    
    # Value sütununu sayısal değere çevir
    filtered_df["value"] = pd.to_numeric(filtered_df["value"], errors='coerce').fillna(0)
    
    # 1. Tekilleştirme
    deduplicated_df = filtered_df.sort_values(by=["category_name", "query", "value"], ascending=[True, True, False]) \
                                 .drop_duplicates(subset=["category_name", "query"], keep="first")

    # 2. ALFABE KONTROLÜ (Sadece Latin Alfabesi)
    import re
    
    def is_latin_alphabet(text):
        if not isinstance(text, str) or text.strip() == "":
            return False
        
        # Sadece Latin alfabesi karakterleri, sayılar, boşluklar ve yaygın noktalama işaretleri
        # Bu pattern Arapça, Çince, Kiril, Japonca vb. latin dışı alfabeleri eler.
        # Latin Extended (Örn: é, ö, ş) gibi karakterleri de dahil etmek için [A-Za-zÀ-ÿ] kullanıyoruz.
        latin_pattern = re.compile(r'^[A-Za-z0-9\s\.,!\?\-\(\)\'\"&/@#\$%]+$')
        
        # Eğer çok kesin bir İngilizce/Sayı/Simge kontrolü istiyorsanız üsttekini;
        # Ama eğer içinde 'ö' veya 'é' olan latin kökenli kelimeler de kalsın derseniz alttakini kullanabiliriz.
        # Sizin durumunuzda LinkedIn/İngilizce ağırlıklı olduğu için standart ASCII + Yaygın simgeler yeterli olacaktır.
        
        return bool(latin_pattern.match(text))

    print(f"Alfabe kontrolü yapılıyor (Latin dışı karakterler temizleniyor)...")
    
    from tqdm import tqdm
    tqdm.pandas()
    
    is_latin_mask = deduplicated_df["query"].progress_apply(is_latin_alphabet)

    final_df = deduplicated_df[is_latin_mask]

    # Son olarak genel query sıralamasını (A-Z) yap
    sorted_df = final_df.sort_values(by="query", ascending=True)

    # Sonucu kaydet
    sorted_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"İşlem tamamlandı!")
    print(f"Başlangıç: {len(top_df)} | Tekilleştirme Sonrası: {len(deduplicated_df)} | İngilizce Filtresi Sonrası: {len(sorted_df)}")
    print(f"Filtrelenmiş dosya şuraya kaydedildi: {output_path}")

if __name__ == "__main__":
    filter_top_queries()
