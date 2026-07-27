

TEENCODE_DICT = {
    
    # Phủ định
    "ko": "không", "k": "không", "khg": "không", "kg": "không", "k0": "không",
    "laems": "lắm", "fai": "phải",
    
    # Viết tắt phổ biến
    "đc": "được", "dc": "được", "đg": "đang", "dg": "đang",
    "sp": "sản phẩm", "sản phảm": "sản phẩm", "hàg": "hàng",
    "shop": "cửa hàng", "st": "siêu thị", "đắc": "đắt",
    "tl": "trả lời", "rep": "trả lời", "fb": "phản hồi",
    "nv": "nhân viên", "bh": "bảo hành", "đt": "điện thoại", "dt": "điện thoại",
    "clg": "chất lượng", "cx": "cũng", "vs": "với", "HSD": "hạn sử dụng", "KH": "khách hàng",
    "nt": "nhắn tin", "r": "rồi", "hsd": "hạn sử dụng", "tks": "cảm ơn", "v": "vậy",
    
    # Từ lóng / Teencode chỉ cảm xúc, mức độ
    "qá": "quá", "wa": "quá", "óa": "quá", "vại": "vậy", "zậy": "vậy",
    "gút": "tốt", "gut": "tốt", "ok": "tốt", "oke": "tốt", "okay": "tốt", "oks": "tốt",
    "laems": "lắm", "lem": "lắm", "nhìu": "nhiều", "nhiu": "nhiều",
    "iu": "yêu", "xih": "xinh", "đúg": "đúng", "bik": "biết",
    
    # Bổ sung đại từ
    "m": "mình", "mn": "mọi người", "t": "tôi", "e": "em", "a": "anh", "c": "chị",
    
    # Song ngữ
    "Vote": "bình chọn", "ship": "vận chuyển", "shop": "cửa hàng", "thank": "cảm ơn",
    "auth": "hàng thật", "size": "kích thước", "hi": "xin chào", "thanks": "cảm ơn",
    "chat": "trò chuyện", "good": "tốt", "check code": "kiểm tra mã"
}

# Clean text
def clean_text(text):
    txt = text.lower()
    txt = re.sub(r'[^\w\s]', ' ', txt)
    words = txt.split()
    cleaned_words = []
    for w in words:
        mapped_w = TEENCODE_DICT.get(w, w)
        cleaned_words.append(mapped_w)
    return " ".join(cleaned_words)

# metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }