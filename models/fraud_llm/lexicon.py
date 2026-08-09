"""Kazakh / Russian / code-switched linguistic resources.

This module deliberately does **not** classify anything. It answers three
narrow questions about a single clause:

1. **What is being talked about?**  (`Concept` hits — an OTP, a card, money, a
   remote-access app, a bank name, a claimed problem …)
2. **What speech act is it?**  (`SpeechAct` — a request, a prohibition, a
   statement, a question.)
3. **Who is the grammatical target?**  (does the speaker want *the other party*
   to do something?)

Deciding whether the conversation is fraud is the job of
`models.fraud_llm.backends.reference`, which reasons over the *sequence* of
these observations per speaker. "OTP mentioned" is not a verdict — "the caller,
after claiming to be bank security and inventing a loan, used an imperative to
demand the OTP" is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")

#: Kazakh-specific Cyrillic graphemes.
KAZAKH_LETTERS = set("әғқңөұүһі")

#: High-frequency Kazakh function words that contain no Kazakh-specific letters,
#: needed so that plain-Cyrillic Kazakh is not mislabelled as Russian.
KAZAKH_FUNCTION_WORDS = {
    "мен", "сен", "сіз", "ол", "біз", "бар", "жоқ", "және", "бірақ", "үшін",
    "керек", "қажет", "деп", "ма", "ме", "ба", "бе", "па", "пе", "да", "де",
    "та", "те", "не", "қалай", "қашан", "неге", "осы", "бұл", "сол", "менің",
    "сіздің", "оның", "тұр", "жатыр", "болды", "еді", "ғой", "ия", "иә", "жарайды",
    "рахмет", "рақмет", "сәлем", "сәлеметсіз", "қазір", "кейін", "тағы", "тек",
}

#: Russian function words used the same way.
RUSSIAN_FUNCTION_WORDS = {
    "и", "в", "на", "с", "по", "что", "это", "как", "не", "вы", "я", "мы", "он",
    "они", "да", "нет", "но", "если", "чтобы", "сейчас", "сразу", "уже", "ваш",
    "ваша", "ваше", "ваши", "мой", "мне", "вам", "вас", "его", "её", "их", "для",
    "или", "тоже", "очень", "потом", "здравствуйте", "спасибо", "пожалуйста",
}


def canon(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower().replace("ё", "е"))).strip()


def tokens(text: str) -> list[str]:
    return canon(text).split()


def detect_language(text: str) -> str:
    """`kk` | `ru` | `mixed` | `unknown` for one utterance.

    Mixed is the common case in Kazakhstan and is never a risk signal — it is
    detected only so the UI and the evaluator can slice by language.
    """
    toks = tokens(text)
    if not toks:
        return "unknown"
    kk_score = 0
    ru_score = 0
    for tok in toks:
        has_kk_letter = bool(KAZAKH_LETTERS & set(tok))
        if has_kk_letter or tok in KAZAKH_FUNCTION_WORDS:
            kk_score += 1
        elif tok in RUSSIAN_FUNCTION_WORDS:
            ru_score += 1
        elif _KK_SUFFIX.search(tok):
            kk_score += 1
        elif re.search(r"[а-я]", tok):
            ru_score += 1
    if kk_score and ru_score:
        weaker = min(kk_score, ru_score)
        stronger = max(kk_score, ru_score)
        if weaker / (weaker + stronger) >= 0.2:
            return "mixed"
        return "kk" if kk_score > ru_score else "ru"
    if kk_score:
        return "kk"
    if ru_score:
        return "ru"
    return "unknown"


#: Kazakh morphology hints (possessive / verbal / case suffixes).
_KK_SUFFIX = re.compile(
    r"(ңыз|ңіз|ыңыз|іңіз|мыз|міз|дың|дің|тың|тің|ның|нің|ларды|лерді|дан|ден|тан|тен|нан|нен|ға|ге|қа|ке|да|де|та|те)$"
)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


class Concept(str, Enum):
    # --- organisations & authority (never evidence on their own) ---
    ORG_BANK = "ORG_BANK"
    ORG_POLICE = "ORG_POLICE"
    ORG_INVESTIGATOR = "ORG_INVESTIGATOR"
    ORG_GOVERNMENT = "ORG_GOVERNMENT"
    ORG_TELCO = "ORG_TELCO"
    ORG_DELIVERY = "ORG_DELIVERY"
    ORG_MARKETPLACE = "ORG_MARKETPLACE"
    ORG_CRYPTO = "ORG_CRYPTO"
    ORG_INVESTMENT = "ORG_INVESTMENT"
    ORG_EMPLOYER = "ORG_EMPLOYER"
    SECURITY_DEPT = "SECURITY_DEPT"

    # --- identity framing ---
    SELF_INTRO = "SELF_INTRO"
    ROLE_CLAIM = "ROLE_CLAIM"

    # --- claimed problems ---
    PROBLEM_LOAN = "PROBLEM_LOAN"
    PROBLEM_TRANSACTION = "PROBLEM_TRANSACTION"
    PROBLEM_BLOCKED = "PROBLEM_BLOCKED"
    PROBLEM_COMPROMISE = "PROBLEM_COMPROMISE"
    PROBLEM_LEGAL = "PROBLEM_LEGAL"
    PROBLEM_REFUND = "PROBLEM_REFUND"

    # --- pressure ---
    URGENCY = "URGENCY"
    THREAT = "THREAT"
    FEAR = "FEAR"
    SECRECY = "SECRECY"
    ISOLATION = "ISOLATION"
    SOLUTION_FRAME = "SOLUTION_FRAME"
    TRUST_FRAME = "TRUST_FRAME"
    REWARD_FRAME = "REWARD_FRAME"

    # --- request targets ---
    TARGET_OTP = "TARGET_OTP"
    TARGET_CODE_GENERIC = "TARGET_CODE_GENERIC"
    TARGET_CODE_WORD = "TARGET_CODE_WORD"
    TARGET_CARD_FULL = "TARGET_CARD_FULL"
    TARGET_CARD_PARTIAL = "TARGET_CARD_PARTIAL"
    TARGET_PASSWORD = "TARGET_PASSWORD"
    TARGET_PERSONAL_DATA = "TARGET_PERSONAL_DATA"
    TARGET_MONEY = "TARGET_MONEY"
    TARGET_SAFE_ACCOUNT = "TARGET_SAFE_ACCOUNT"
    TARGET_CASH_POINT = "TARGET_CASH_POINT"
    TARGET_REMOTE_APP = "TARGET_REMOTE_APP"
    TARGET_REMOTE_ACCESS = "TARGET_REMOTE_ACCESS"
    TARGET_SCREEN = "TARGET_SCREEN"
    TARGET_APP_INSTALL = "TARGET_APP_INSTALL"
    TARGET_LINK = "TARGET_LINK"
    TARGET_CRYPTO = "TARGET_CRYPTO"

    # --- legitimacy signals ---
    OFFICIAL_CHANNEL = "OFFICIAL_CHANNEL"
    NEVER_ASKS = "NEVER_ASKS"
    WARNING_FRAME = "WARNING_FRAME"
    CUSTOMER_INQUIRY = "CUSTOMER_INQUIRY"
    VICTIM_REFUSAL = "VICTIM_REFUSAL"
    VICTIM_COMPLIANCE = "VICTIM_COMPLIANCE"
    VICTIM_DOUBT = "VICTIM_DOUBT"


#: concept -> substring stems, matched against canonicalised text.
#: Stems are chosen to survive Kazakh agglutination ("аудар" covers
#: "аударыңыз / аударасыз / аударуыңыз керек") and Russian inflection.
LEXICON: dict[Concept, tuple[str, ...]] = {
    Concept.ORG_BANK: (
        "банк", "kaspi", "каспи", "каспий", "halyk", "халык", "халық", "халiк",
        "forte", "форте", "jusan", "жусан", "жүсан", "freedom", "фридом",
        "центркредит", "bcc", "отбасы", "нурбанк", "нұрбанк", "евразийск",
        "eurasian", "altyn", "алтын банк", "home credit", "хоум кредит",
        "ұлттық банк", "нацбанк", "национальн банк", "bereke", "береке",
    ),
    Concept.ORG_POLICE: (
        "полиц", "полициянын", "участков", "мвд", "ішкі істер", "кнб", "ұқк",
        "финпол", "антикор", "департамент полиции",
    ),
    Concept.ORG_INVESTIGATOR: (
        "следовател", "следствен", "тергеуші", "тергеу", "прокурат", "прокурор",
        "оперуполномоченн", "дознавател", "судебн пристав", "сот орындаушы",
    ),
    Concept.ORG_GOVERNMENT: (
        "egov", "егов", "егоб", "цон", "хцон", "гос услуг", "госуслуг",
        "министерств", "әкімдік", "акимат", "налогов", "салық", "енпф",
        "соцстрах", "әлеуметтік қор", "суд", "сот",
    ),
    Concept.ORG_TELCO: (
        "beeline", "билайн", "activ", "актив", "kcell", "кселл", "tele2",
        "теле2", "altel", "алтел", "izi", "мобильн оператор", "ұялы оператор",
    ),
    Concept.ORG_DELIVERY: (
        "курьер", "курьерск", "доставк", "жеткіз", "посылк", "сәлемдеме",
        "cdek", "сдэк", "казпочт", "қазпошт", "wolt", "glovo", "жеткізу қызметі",
    ),
    Concept.ORG_MARKETPLACE: (
        "olx", "олх", "satu", "сату кз", "wildberries", "вайлдберриз", "ozon",
        "озон", "объявлен", "хабарланд", "маркетплейс", "kaspi магазин",
    ),
    Concept.ORG_CRYPTO: (
        "крипт", "биткоин", "bitcoin", "btc", "usdt", "binance", "бинанс",
        "bybit", "байбит", "эфириум", "ethereum", "обменник", "кошелек",
        "әмиян", "токен",
    ),
    Concept.ORG_INVESTMENT: (
        "инвест", "трейд", "брокер", "форекс", "forex", "акци", "дивиденд",
        "портфель", "доходность", "пассивн доход", "инвестициял",
    ),
    Concept.ORG_EMPLOYER: (
        "ваканси", "работодател", "hr отдел", "жұмыс беруші", "жұмысқа",
        "подработк", "удаленн работ", "қосымша жұмыс",
    ),
    Concept.SECURITY_DEPT: (
        "служба безопасност", "отдел безопасност", "қауіпсіздік қызмет",
        "қауіпсіздік бөлім", "финансов мониторинг", "антифрод", "фрод монитор",
        "департамент безопасност", "қауіпсіздік департамент",
    ),
    Concept.SELF_INTRO: (
        "меня зовут", "менің атым", "вас беспокоит", "сізді алаңдатып",
        "это ", "бұл ", "звоню из", "звоню вам из", "я из", "мен мұнда",
        "қоңырау соғып отырмын",
        "хабарласып отырмын", "мен сізге", "я звоню", "на связи",
    ),
    Concept.ROLE_CLAIM: (
        "я сотрудник", "сотрудник банка", "я специалист", "специалист отдела",
        "я представляю", "я менеджер", "мен қызметкер", "қызметкері", "маман",
        "техподдержк", "тех поддержк", "служба поддержк", "отдел поддержк",
        "тех қолдау", "қолдау көрсету қызмет",
        "мен банк", "старший следователь", "капитан полиции", "майор полиции",
        "оперуполномоченный", "я работаю в", "мен жұмыс жасаймын", "менеджер",
        "консультант", "инспектор", "агент", "отдел подбора", "подбор персонала",
        "рекрутер", "hr менеджер", "hr отдел", "hr бөлім", "кадр бөлім", "персонал таңдау",
    ),
    Concept.PROBLEM_LOAN: (
        "оформляется кредит", "оформлен кредит", "оформили кредит",
        "заявка на кредит", "заявку на кредит", "кредит на ваше имя",
        "микрозайм", "займ на ваше имя", "несие рәсімдел", "кредит рәсімдел",
        "несие алын", "сізге кредит", "кредит өтінім", "несие өтінім",
        "рассрочк на ваше имя",
    ),
    Concept.PROBLEM_TRANSACTION: (
        "подозрительн операц", "подозрительн транзакц", "подозрительн платеж",
        "несанкционированн", "попытка списания", "попытка перевода",
        "списание средств", "списать деньги", "күдікті операц", "күдікті аударым",
        "рұқсатсыз операц", "рұқсат етілмеген", "странн операц", "операция түсті",
        "перевод на неизвестн", "танылмаған төлем", "заявка на перевыпуск",
        "перевыпуск", "қайта шығаруға өтінім",
    ),
    Concept.PROBLEM_BLOCKED: (
        "заблокирован", "блокиров", "будет заблокирован", "блокта", "бұғатта",
        "бұғаттал", "шот жабыл", "счет будет закрыт", "карта не работает",
        "доступ приостановлен", "приостановлен",
    ),
    Concept.PROBLEM_COMPROMISE: (
        "взломан", "скомпрометирован", "утечк данных", "мошенники получили",
        "третьи лица получили доступ", "доступ к вашему счету", "хакер",
        "деректер ұрлан", "алаяқтар қол жеткіз", "шотыңызға кірген",
        "ваши данные попали", "мәліметтеріңіз таралған",
    ),
    Concept.PROBLEM_LEGAL: (
        "уголовн дело", "қылмыстық іс", "административн", "штраф", "айыппұл",
        "задолженност", "берешек", "исполнительн производств", "повестк",
        "против вас", "сізге қатысты іс", "розыск", "іздестір",
    ),
    Concept.PROBLEM_REFUND: (
        "возврат средств", "компенсац", "переплат", "вам положен возврат",
        "қаражат қайтар", "өтемақы", "ақшаңызды қайтар", "возместим",
    ),
    Concept.URGENCY: (
        "срочно", "немедленно", "прямо сейчас", "сейчас же", "как можно быстрее",
        "в течение", "минут", "не теряйте время", "быстрее", "успеть",
        "пока не поздно", "времени мало", "жедел", "дереу", "тез", "асығ",
        "уақыт жоқ", "кешіктірмеңіз", "қазір қазір", "сол сәтте", "шұғыл",
    ),
    Concept.THREAT: (
        "будет заблокирован", "заблокируем", "спишут", "будет списан",
        "потеряете", "лишитес", "возбудим", "привлеч к ответственност",
        "уголовн ответственност", "арест", "суд", "жауапкершілік",
        "айырылас", "жоғалт", "тұтқында", "сотқа бер", "бұғаттаймыз",
    ),
    Concept.FEAR: (
        # NB: stems must not match "безопасность" / "қауіпсіздік" — the safety
        # words that appear in every legitimate bank greeting. Hence the
        # multi-word forms rather than a bare "опасност" / "қауіп".
        "в опасност", "под угроз", "есть угроза", "угрожа", "угрозы",
        "мошенник", "алаяқтар", "алаяқтық", "қауіпте", "қауіп бар", "қауіп төн",
        "рискуете", "тәуекел", "можете потерять", "потеряете все",
    ),
    Concept.SECRECY: (
        "никому не говорите", "никому не сообщайте", "не рассказывайте",
        "конфиденциальн", "тайна следствия", "между нами", "это секрет",
        "ешкімге айтпаңыз", "құпия", "айтпаңыз ешкімге", "жасырын",
    ),
    Concept.ISOLATION: (
        "не отключайтесь", "оставайтесь на линии", "не сбрасывайте",
        "не звоните в банк", "не обращайтесь в отделение", "не выходите",
        "не кладите трубку", "телефонды тастамаңыз", "байланысты үзбеңіз",
        "желіде болыңыз", "банкке қоңырау соқпаңыз", "бөлімшеге барма",
    ),
    Concept.SOLUTION_FRAME: (
        "чтобы отменить", "для отмены", "чтобы остановить", "чтобы защитить",
        "чтобы сохранить", "единственн способ", "мы поможем", "мы решим",
        "нужно защитить", "спасем ваши деньги", "болдырмау үшін",
        "тоқтату үшін", "қорғау үшін", "сақтау үшін", "жалғыз жол",
        "көмектесем", "шешеміз", "бас тарту үшін",
    ),
    Concept.TRUST_FRAME: (
        "я вам помогу", "не волнуйтес", "все под контролем", "доверьтес",
        "мы на вашей стороне", "уверяю", "алаңдама", "сенсеңіз болады",
        "мен көмектесемін", "бәрі жақсы болады",
    ),
    Concept.REWARD_FRAME: (
        "гарантированн доход", "заработ", "бонус", "приз", "выигр",
        "кешбек", "кэшбек", "табыс", "жеңіп ал", "сыйлық", "легк деньги",
        "доход в месяц", "процент в день",
    ),
    Concept.TARGET_OTP: (
        "код из sms", "код из смс", "смс код", "sms код", "смс кода",
        "код подтвержден", "одноразов код", "код который приде", "код придет",
        "четырехзначн код", "шестизначн код", "пароль из смс", "код из сообщени",
        "растау код", "бір реттік код", "смс келеді", "смс кел", "кодты",
        "код келеді", "смстегі код", "хабарламадағы код", "push код",
        "код из push", "код из приложени",
    ),
    Concept.TARGET_CODE_GENERIC: ("код", "коды", "кодом", "кодты", "кодын"),
    Concept.TARGET_CODE_WORD: (
        # A standing secret the customer chose, used for identification. Banks
        # legitimately ask for it; it is not a one-time transaction code.
        "кодов слов", "кодовое слово", "кодтық сөз", "код сөз", "секретн слов",
        "контрольн слов",
    ),
    Concept.TARGET_CARD_FULL: (
        "номер карт", "номер вашей карт", "полный номер карт", "cvv", "cvc",
        "три цифры", "трехзначн код на обороте", "срок действия карт",
        "карта нөмір", "картаңыздың нөмір", "картаның нөмір", "16 цифр",
        "данные карт", "карта деректер", "реквизиты карт",
    ),
    Concept.TARGET_CARD_PARTIAL: (
        "последние четыре", "последние 4", "четыре последн", "соңғы төрт",
        "соңғы 4", "последние цифры карт",
    ),
    Concept.TARGET_PASSWORD: (
        "пароль", "логин", "пин код", "пин кода", "пинкод", "код доступа",
        "құпия сөз", "парол", "пароль от личного кабинета", "от приложени пароль",
    ),
    Concept.TARGET_PERSONAL_DATA: (
        "иин", "жсн", "номер удостоверен", "удостоверен личност", "жеке куалік",
        "жеке куәлік", "паспорт", "дата рожден", "туған күн", "фио",
        "ваши полные данн", "толық аты жөні", "адрес прописк",
    ),
    Concept.TARGET_MONEY: (
        "переведите", "перевести", "перевод", "перечисл", "внесите",
        "положите на", "снимите деньги", "снять деньги", "аудар", "ақша сал",
        "салыңыз", "ақшаны сал",
        "ақшаны жіберіңіз", "қаражатты аудар", "сумму", "сомма", "сома",
        "тенге", "теңге", "миллион", "тысяч", "мың",
    ),
    Concept.TARGET_SAFE_ACCOUNT: (
        "безопасн счет", "безопасн счёт", "защищенн счет", "страхов счет",
        "резервн счет", "специальн счет", "транзитн счет", "безопасн кошелек",
        "қауіпсіз шот", "сақтық шот", "арнайы шот", "уақытша шот",
        "ячейк", "депозитн ячейк",
    ),
    Concept.TARGET_CASH_POINT: (
        "банкомат", "терминал", "касс", "отделени рядом", "банкоматқа",
        "терминалға", "qr код на банкомате",
    ),
    Concept.TARGET_REMOTE_APP: (
        "anydesk", "энидеск", "эни деск", "teamviewer", "тимвьювер",
        "rustdesk", "quick support", "quicksupport", "airdroid", "aweray",
        "supremo", "ammyy",
    ),
    Concept.TARGET_REMOTE_ACCESS: (
        "удаленн доступ", "удаленно подключ", "подключимся к вашему телефону",
        "дистанционн доступ", "қашықтан қосыл", "қашықтан кір",
        "телефоныңызға қосыл", "дам вам код доступа к устройству",
    ),
    Concept.TARGET_SCREEN: (
        "демонстрац экран", "покажите экран", "поделитес экраном",
        "включите демонстрац", "трансляц экран", "экранды көрсет",
        "экранды бөл", "экран көрсет", "экранды бөліс", "экранды ұсын",
    ),
    Concept.TARGET_APP_INSTALL: (
        "установите приложени", "скачайте приложени", "загрузите приложени",
        "установите программ", "apk", "по ссылке установите", "қолданба орнат",
        "қолданбаны жүктеп", "программа орнат", "жүктеп алыңыз",
        "установите наше приложени",
    ),
    Concept.TARGET_LINK: (
        "по ссылке", "ссылку отправл", "перейдите по", "нажмите на ссылк",
        "сілтеме", "сілтемеге кір", "ссылка в смс",
    ),
    Concept.TARGET_CRYPTO: (
        "usdt", "биткоин", "крипто", "крипт кошелек", "пополните кошелек",
        "трейдинг счет", "инвестиционн счет", "әмиянды толтыр", "депозит",
    ),
    Concept.OFFICIAL_CHANNEL: (
        "перезвоните по номеру на обратной стороне", "официальн номер",
        "обратитес в отделени", "через приложени банка", "в мобильном приложени",
        "позвоните сами", "бөлімшеге барыңыз", "қосымша арқылы", "1414",
        "официальн сайт", "напишите в чат приложени", "call центр",
        "колл центр", "горяч лини",
    ),
    Concept.NEVER_ASKS: (
        "никогда не спрашивает", "не спрашиваем код", "мы не запрашиваем",
        "сотрудник банка никогда", "банк никогда не", "ешқашан сұрамайды",
        "біз код сұрамаймыз", "сұрамаймыз", "не имеем права спрашивать",
        "нам не нужен ваш код",
    ),
    Concept.WARNING_FRAME: (
        # The speaker is *describing* a scammer's behaviour, not performing it:
        # "if someone calls you and asks you to transfer money, that's a scam".
        "если вам звонят", "если позвонят", "если просят", "если попросят",
        "если вас просят", "звонят и просят", "это мошенники", "будут мошенники",
        "мошенники просят", "мошенники звонят", "под видом банка", "представляются банком",
        "егер қоңырау соқса", "егер сұраса", "бұл алаяқтар", "алаяқтар сұрайды",
        "алаяқтар қоңырау", "банк болып таныст",
    ),
    Concept.CUSTOMER_INQUIRY: (
        "я хотел узнать", "хотела узнать", "подскажите", "у меня вопрос",
        "я звоню по поводу", "хочу уточнить", "мен білгім кел", "сұрағым бар",
        "айтып жіберіңізші", "как мне", "қалай жасай", "не могу", "почему",
        "неге", "жасалмады", "не прошел", "не прошла", "өтпей жатыр",
    ),
    Concept.VICTIM_REFUSAL: (
        "я не буду", "не скажу", "не назову", "я вам не верю", "это мошенник",
        "я перезвоню сам", "я схожу в отделени", "айтпаймын", "сенбеймін",
        "мен өзім хабарласам", "не собираюсь", "положу трубку",
    ),
    Concept.VICTIM_COMPLIANCE: (
        "хорошо сейчас", "уже перевожу", "перевела", "перевел", "диктую",
        "код такой", "сейчас скажу", "устанавливаю", "скачал", "скачала",
        "жарайды қазір", "аударып жатырмын", "аудардым", "орнаттым",
        "айтамын", "жіберемін", "код келді",
    ),
    Concept.VICTIM_DOUBT: (
        "я ничего не оформлял", "я не оформляла", "но я не", "я не заказывал",
        "странно", "мен ондай", "мен рәсімдеген жоқпын", "тапсырыс бермедім",
        "а откуда вы", "как я могу вам верить", "сіз кімсіз", "точно",
        "не понимаю", "түсінмедім",
    ),
}


# ---------------------------------------------------------------------------
# Speech acts
# ---------------------------------------------------------------------------


class SpeechAct(str, Enum):
    REQUEST = "REQUEST"          # "назовите код", "аударыңыз"
    PROHIBITION = "PROHIBITION"  # "не сообщайте код", "айтпаңыз"
    QUESTION = "QUESTION"
    STATEMENT = "STATEMENT"


#: Russian directive verbs (imperative or "вы должны/нужно" frames).
RU_REQUEST_MARKERS = (
    "назовите", "назови", "скажите", "скажи", "продиктуйте", "продиктуй", "диктуйте",
    "сообщите", "сообщи", "введите", "введи", "отправьте", "перешлите",
    "переведите", "перечислите", "снимите", "внесите", "положите",
    "установите", "скачайте", "загрузите", "откройте", "нажмите", "зайдите",
    "подтвердите", "поделитесь", "включите", "продублируйте", "повторите",
    "дайте", "укажите", "оформите", "пополните", "переходите", "перейдите",
    "оставайтесь", "подойдите", "подключите", "разрешите", "напишите",
    "мне нужен", "мне нужно чтобы вы", "нужно назвать", "нужно перевести",
    "нужно сообщить", "нужно установить", "надо перевести", "надо назвать",
    "вы должны", "вам необходимо", "вам нужно", "потребуется сообщить",
    "прошу назвать", "прошу продиктовать", "жду код", "давайте код",
    "какой код", "какой у вас код", "назовите пожалуйста",
)

#: Kazakh directives: explicit forms plus the polite imperative suffix.
KK_REQUEST_MARKERS = (
    "айтыңыз", "айтып жіберіңіз", "айтыңызшы", "жіберіңіз", "аударыңыз",
    "аударып жіберіңіз", "орнатыңыз", "жүктеңіз", "ашыңыз", "басыңыз",
    "көрсетіңіз", "растаңыз", "енгізіңіз", "салыңыз", "беріңіз", "барыңыз",
    "күтіңіз", "тұрыңыз", "қосыңыз", "толтырыңыз", "оқыңыз", "аудару керек",
    "айту керек", "жіберу керек", "орнату қажет", "аудару қажет",
    "айту қажет", "аударуыңыз керек", "айтуыңыз керек", "не код келді",
    "кодты айт",
)

#: Negated necessity: the speaker says the action is *not* required.
NEGATED_NECESSITY = (
    "не нужно", "не требуется", "не обязательно", "не надо", "не запрашиваем",
    "не спрашиваем", "не имеем права", "қажет емес", "керек емес", "сұрамаймыз",
    "ешқашан сұрамайды", "сұрамайды",
)

#: Explicit negative-imperative forms (Russian) — protective advice.
RU_PROHIBITION_MARKERS = (
    "не говорите", "не сообщайте", "не называйте", "не диктуйте",
    "не переводите", "не устанавливайте", "не скачивайте", "не переходите",
    "не нажимайте", "не давайте", "не показывайте", "никогда не сообщайте",
    "никому не сообщайте", "ни в коем случае не", "не вводите", "не делитесь",
    # "код мне называть не нужно" — a refusal to take a secret, not a request.
    "не нужно", "не требуется", "не обязательно", "не запрашиваем", "не спрашиваем",
)

#: Kazakh negative imperative: stem + (па|пе|ба|бе|ма|ме) + (ңыз|ңіз).
_KK_PROHIBITION = re.compile(r"\w+(па|пе|ба|бе|ма|ме)(ңыз|ңіз|ңдар|ңдер|ңыздар|ңіздер)?\b")
KK_PROHIBITION_MARKERS = (
    "айтпаңыз", "хабарламаңыз", "бермеңіз", "аудармаңыз", "орнатпаңыз",
    "баспаңыз", "кірмеңіз", "жүктемеңіз", "көрсетпеңіз", "сенбеңіз",
)

_QUESTION_WORDS = (
    "как", "почему", "зачем", "когда", "какой", "какая", "сколько", "где",
    "кто", "что делать", "қалай", "неге", "қашан", "қанша", "кім", "не үшін",
)

_CLAUSE_SPLIT = re.compile(
    r"[.!?;\n]+"
    r"|,\s+(?=и |а |но |чтобы |потом |затем |только |затем )"
    r"|,\s+(?=тек |және |бірақ |содан |сосын |сондықтан |әйтпесе )"
)


@dataclass
class Clause:
    """One clause with its detected concepts and speech act."""

    text: str
    concepts: set[Concept] = field(default_factory=set)
    act: SpeechAct = SpeechAct.STATEMENT
    matched: dict[Concept, str] = field(default_factory=dict)

    def has(self, *concepts: Concept) -> bool:
        return any(c in self.concepts for c in concepts)


def split_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(text or "") if p and p.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


# ---------------------------------------------------------------------------
# Stem matching
# ---------------------------------------------------------------------------
#
# Russian inflects and Kazakh agglutinates, so a stem has to match a *word
# prefix*, not an arbitrary substring. Two rules, applied per token:
#
#   * tokens of 4+ characters match as a word prefix  ("аудар" → "аударыңыз",
#     "подозрительн" → "подозрительная")
#   * tokens of 3 or fewer characters must match the whole word ("сот" must not
#     fire on "сотрудник"; "код" must not fire on "кодовое слово")
#
# Anchoring at a word start is also what stops "безопасность" — present in every
# legitimate bank greeting — from matching a fear stem like "опасност".
#
#: Concepts whose stems must always match whole words, however long.
_EXACT_CONCEPTS: frozenset[Concept] = frozenset({Concept.TARGET_CODE_GENERIC})


def _compile_stem(stem: str, *, exact: bool) -> re.Pattern[str] | None:
    probe = canon(stem)
    if not probe:
        return None
    words = probe.split()
    parts: list[str] = []
    for index, token in enumerate(words):
        escaped = re.escape(token)
        is_last = index == len(words) - 1
        # Inflection lands on the final word of a phrase ("қауіпсіз шотқа",
        # "ақша салыңыз"), so the last token of a multi-word stem always matches
        # as a prefix. A short *single-word* stem stays exact, which is what
        # keeps "код" off "кодовое слово" and "сот" off "сотрудник".
        prefix_ok = not exact and (len(token) >= 4 or (is_last and len(words) > 1))
        parts.append(rf"{escaped}\w*" if prefix_ok else rf"{escaped}(?!\w)")
    return re.compile(r"(?<!\w)" + r"\s+".join(parts))


#: concept -> [(original stem, compiled pattern)], built once at import.
_PATTERNS: dict[Concept, tuple[tuple[str, re.Pattern[str]], ...]] = {
    concept: tuple(
            (stem, pattern)
            for stem in stems
            if (pattern := _compile_stem(stem, exact=concept in _EXACT_CONCEPTS)) is not None
        )
    for concept, stems in LEXICON.items()
}


def find_concepts(text: str) -> dict[Concept, str]:
    """Concept -> the exact stem that matched (kept for auditability)."""
    hay = canon(text)
    if not hay:
        return {}
    found: dict[Concept, str] = {}
    for concept, patterns in _PATTERNS.items():
        for stem, pattern in patterns:
            if pattern.search(hay):
                found[concept] = stem
                break
    return _resolve_overlaps(found, hay)


def _resolve_overlaps(found: dict[Concept, str], hay: str) -> dict[Concept, str]:
    """Keep the most specific concept when several fire on the same words."""
    if Concept.TARGET_OTP in found:
        found.pop(Concept.TARGET_CODE_GENERIC, None)
    if Concept.TARGET_CARD_FULL in found:
        found.pop(Concept.TARGET_CARD_PARTIAL, None)
    # "пароль из смс" is an OTP, not an account password.
    if Concept.TARGET_OTP in found and "пароль из смс" in hay:
        found.pop(Concept.TARGET_PASSWORD, None)
    # A generic "код" next to an app/lock context is not necessarily an OTP.
    return found


def classify_act(text: str) -> SpeechAct:
    """Request / prohibition / question / statement for one clause."""
    hay = canon(text)
    if not hay:
        return SpeechAct.STATEMENT
    padded = f" {hay} "

    # "код айту қажет емес" / "код называть не нужно" is a refusal to take a
    # secret. It contains a necessity marker, so it must be checked before the
    # request markers or it reads as a demand.
    if any(canon(marker) in hay for marker in NEGATED_NECESSITY):
        return SpeechAct.PROHIBITION

    forbids = any(canon(marker) in hay for marker in RU_PROHIBITION_MARKERS + KK_PROHIBITION_MARKERS)
    if not forbids:
        # Kazakh negative imperative found morphologically.
        forbids = any(
            len(tok) >= 6 and _KK_PROHIBITION.fullmatch(tok) for tok in hay.split()
        )
    demands = any(
        canon(marker) in hay for marker in RU_REQUEST_MARKERS + KK_REQUEST_MARKERS
    )
    # A clause carrying both ("енгізіңіз, тек көрсетуді жаппаңыз" / "назовите код,
    # но никому не говорите") is a demand with a side condition, not advice.
    # Reading it as a prohibition is how an attack gets mistaken for a warning.
    if demands:
        return SpeechAct.REQUEST
    if forbids:
        return SpeechAct.PROHIBITION

    if "?" in (text or ""):
        for qw in _QUESTION_WORDS:
            if canon(qw) in hay:
                return SpeechAct.QUESTION
        return SpeechAct.QUESTION
    for qw in _QUESTION_WORDS:
        if padded.startswith(f" {canon(qw)}"):
            return SpeechAct.QUESTION
    return SpeechAct.STATEMENT


def analyse_clauses(text: str) -> list[Clause]:
    """Full per-clause concept + speech-act analysis of one utterance."""
    out: list[Clause] = []
    for raw in split_clauses(text):
        matched = find_concepts(raw)
        out.append(
            Clause(
                text=raw,
                concepts=set(matched),
                act=classify_act(raw),
                matched=matched,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Sensitive-data masking (spec section 27)
# ---------------------------------------------------------------------------

_DIGIT_RUNS = re.compile(r"(?<!\d)(\d[\d\s-]{1,22}\d)(?!\d)")
_CREDENTIAL_CUE = re.compile(
    r"(код|code|kod|otp|пароль|парол|пин|pin|cvv|cvc|карт|karta|иин|жсн|құпия)",
    re.IGNORECASE,
)
#: A number followed by a currency is an amount — evidence, not a secret.
_CURRENCY_AFTER = re.compile(
    r"^\s*(тенге|теңге|тг|доллар|dollars?|usd|евро|eur|рубл|сом|миллион|мың|тысяч)",
    re.IGNORECASE,
)


def mask_sensitive(text: str) -> tuple[str, list[str]]:
    """Mask OTPs / card numbers / PINs for long-term storage.

    Returns `(masked_text, list_of_mask_labels)`. Only digit runs that appear in
    a credential context are masked, so amounts like "3 000 000 тенге" survive —
    they are evidence, not secrets.
    """
    if not text:
        return text, []
    labels: list[str] = []

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        digits = re.sub(r"\D", "", raw)
        # Only the words immediately before the number can make it a credential;
        # a wide window lets an unrelated "код" earlier in the sentence mask an
        # amount, and amounts are evidence.
        preceding = text[max(0, match.start() - 28) : match.start()]
        following = text[match.end() : match.end() + 14]
        looks_like_card = len(digits) >= 12
        if _CURRENCY_AFTER.match(following) and not looks_like_card:
            return raw
        cued = bool(_CREDENTIAL_CUE.search(preceding))
        if looks_like_card:
            labels.append("CARD_NUMBER")
            return "«[CARD_MASKED]»"
        if cued and 3 <= len(digits) <= 8:
            labels.append("OTP_OR_PIN")
            return "«[CODE_MASKED]»"
        return raw

    masked = _DIGIT_RUNS.sub(repl, text)
    return masked, labels
