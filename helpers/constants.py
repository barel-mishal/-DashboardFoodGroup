ISRAEL_OFFICIAL_DATA_PATH = 'csvs/original_israeli_data.csv'
ISRAELI_DATA_PATH = "csvs/israeli_data.csv"
FDC_DATA_PATH = "csvs/wide_nutri_records.csv"
ISRAELI_LABELED_PATH = "csvs/labeled_data.xlsx"

NUMBER_OF_FOOD_GROUPS = 32

MACRO_NUTRIENTS = ['protein', 'total_fat', 'carbohydrates']
MICRO_MINERALS = ['calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'sodium', 'zinc', 'copper']
MICRO_VITAMINS = ['vitamin_a_iu', 'vitamin_e', 'vitamin_c', 'thiamin', 'riboflavin', 'niacin', 'vitamin_b6',
                  'folate', 'folate_dfe', 'vitamin_b12', 'carotene']
MICRO_NUTRIENTS = MICRO_MINERALS + MICRO_VITAMINS

FDC_MACRO = ['Protein', 'Total lipid (fat)', 'Carbohydrate, by difference']
FDC_MINERALS = ['Calcium, Ca', 'Iron, Fe', 'Magnesium, Mg', 'Phosphorus, P', 'Potassium, K', 'Sodium, Na', 'Zinc, Zn',
                'Copper, Cu']
FDC_VITAMINS = ['Vitamin A, IU', 'Vitamin E (alpha-tocopherol)', 'Vitamin C, total ascorbic acid', 'Thiamin',
                'Riboflavin', 'Niacin', 'Vitamin B-6', 'Folate, food', 'Folate, DFE', 'Vitamin B-12', 'Carotene, alpha']
FDC_MICRO = FDC_MINERALS + FDC_VITAMINS

FILTER = 'מלבין קפה|תחליף|אבקה|תחליף נוזלי|תחליף לארוחה|תחליף או תוסף|תחליף או תוספת|ביצה חלבון מיובש|סויה, חלבון סויה|תמ"י|גלוטן חיטה, יבש|תבלינים|רימון חי, עם קליפה|תמצית|תמ"ל|אבקה' + 'מי גבינה|איזיליין|תמ"י|תחליף|WITH FIBER|BENEFIBER|תמ"ל|לפגים|מטרנה|איזומיל|סימילאק|אבקת|אומגה|באומגה|אנשור|לתינוק|פסולת|whey|פגים'
FOOD_STOP_WORDS = ['עם', 'אחר', 'ללא', 'לא', 'על', 'בשומן', 'בשמן', 'או', 'מבושל', 'מבושלים', 'מבושלת', 'מבושלות']

LABELS_DIC = {
    56208068:	'דגנים מלאים',
    51140668:	'דברי מאפה (לחמים) מדגן מלא',
    56204930:	'דגנים לא מלאים',
    51101009:	'דברי מאפה (לחמים) מדגן לא מלא',
    71001040:	'ירקות עמילניים',
    58126148:	'דברי מאפה מלוחים',
    53510019:	'דברי מאפה מתוקים',
    57208099:	'מידגנים (Cereals)',
    92530229:	'משקאות ממותקים בסוכר',
    64105419:	'משקאות ממותקים בתחליפי סוכר',
    92205000:	'משקאות אורז/ שיבולת שועל',
    93504000:	'משקאות אלכוהוליים לא מתוקים',
    93401010:	'משקאות אלכוהוליים שמכילים פחמימות',
    75100780:	'קבוצת הירקות',
    63149010:	'קבוצת הפירות',
    41101219:	'קבוצת קטניות',
    42116000:	'קבוצת אגוזים וגרעינים',
    43103339:	'קבוצת שמנים צמחיים',
    83107000:	'שומנים מהחי',
    90000007:	'מוצרי חלב לא ממותקים',
    11411609:	'מוצרי חלב ממותקים בסוכר',
    11511269:	'קבוצת מוצרי חלב ממותקים בתחליפי סוכר- כל הדיאט',
    41420010:	'קבוצת תחליפי חלב על בסיס קטניות- מוצרי סויה',
    24102010:  'קבוצת הבשר - עוף הודו',
    26115108:	'קבוצת הבשר - דגים',
    23200109:	'קבוצת הבשר - בשר בקר וצאן',
    50030109:	'קבוצת הבשר - תחליפי בשר מהצומח',
    91703070:	'קבוצת הסוכרים',
    31104000:	'ביצים',
    14210079:	'מוצרי חלב דל שומן',
    95312600:	'משקה אנרגיה',
    41811939:	'תחליפי בשר (לייט)'
}
