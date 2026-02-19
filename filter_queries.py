import pandas as pd
from pathlib import Path

def filter_and_sort_queries():
    # Dosya yolları
    input_path = Path("output/related_queries.csv")
    output_path = Path("output/filtered_queries.csv")

    if not input_path.exists():
        print(f"Hata: {input_path} dosyası bulunamadı!")
        return

    print(f"'{input_path}' okunuyor...")
    
    # CSV dosyasını oku
    df = pd.read_csv(input_path)

    # İstenen sütunları seç
    # category_name, type, query, value
    selected_columns = ["category_name", "type", "query", "value"]
    
    # Sütunların varlığını kontrol et
    missing_cols = [col for col in selected_columns if col not in df.columns]
    if missing_cols:
        print(f"Hata: Dosyada şu sütunlar eksik: {missing_cols}")
        return

    filtered_df = df[selected_columns]

    # Query sütununa göre A'dan Z'ye sırala
    sorted_df = filtered_df.sort_values(by="query", ascending=True)

    # Sonucu kaydet
    sorted_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"İşlem tamamlandı!")
    print(f"Toplam satır sayısı: {len(sorted_df)}")
    print(f"Filtrelenmiş dosya şuraya kaydedildi: {output_path}")

if __name__ == "__main__":
    filter_and_sort_queries()
