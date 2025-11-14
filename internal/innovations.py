import numpy as np

def find_score_of_dimension(row, window, domain_size):
    """
    محاسبه امتیاز نوسان نرمال‌شده
    (ScoreJ)
    بر اساس فرمول پایان‌نامه.
    
    Args:
        row (list or np.array)
        window(int)
        domain_size (int)
        
    Returns:
        float: مقدار score_j بین 0 و 1
    """
    data = np.array(row[:window])
    
    if len(data) < 2:
        return 0.0
    
    # محاسبه صورت کسر: مجموع قدر مطلق تفاضل‌های متوالی
    # np.diff تفاضل هر عنصر با عنصر بعدی را حساب می‌کند
    absolute_changes = np.abs(np.diff(data))
    sum_changes = np.sum(absolute_changes)
    
    # W تعداد گام‌های تغییر است (تعداد داده‌ها منهای یک)
    w = len(data) - 1
    
    if domain_size <= 0 or w == 0:
        return 0.0
        
    denominator = w * domain_size
    
    score_j = sum_changes / denominator    
    return score_j

def find_discretization_step(score, domain_size, eta):
    return (score * domain_size) / eta
