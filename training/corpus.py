"""Script library + conversation synthesiser for the QORGAU corpus.

Why the corpus is built this way
--------------------------------
Spec sections 13-16 require **full conversations**, not isolated sentences; a
large share of **hard negatives**; natural **code-switching**; and **ASR noise**.
Annotating that by hand for a prototype is not feasible, so the corpus is
generated from annotated *script families*:

* a family is a social-engineering (or legitimate) **pattern**, written as an
  ordered list of turns;
* each turn carries its speaker, conversation stage, and the risk events it
  realises — the annotation lives on the script, not on the sentence;
* rendering picks a Russian, Kazakh or code-switched surface form and optionally
  degrades it with ASR noise.

Two consequences matter. First, evidence is never fabricated: a gold
`risk_factor` quotes the rendered turn it came from, verbatim. Second, splits are
done **by family** (spec section 25), so no test conversation is a paraphrase of
a training one — the evaluation measures generalisation to unseen scripts rather
than memorisation.

Honest limitation: synthetic data written by one author has a narrower
distribution than real calls, and a model trained only on it will overfit to
these phrasings. The pipeline is built so real annotated calls can be dropped
into `datasets/raw/` and mixed in (see `training/prepare_dataset.py`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from config.ontology import (
    Classification,
    EVENT_TO_TACTIC,
    EventCategory as EC,
    HARMFUL_ACTION_EVENTS,
    IMPERSONATION_SCAM_TYPES,
    STAGE_DEPTH,
    ScamType as ST,
    Severity as SV,
    Speaker,
    Stage as SG,
)

# ---------------------------------------------------------------------------
# Script model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    stage: str
    ru: str
    kk: str
    #: Hand-written intra-sentential code-switch. When absent the renderer
    #: alternates languages between turns, which is also natural in Kazakhstan.
    mixed: str | None = None
    events: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    classification: str
    call_direction: str
    turns: tuple[Turn, ...]
    scam_types: tuple[str, ...] = ()
    difficulty: str = "obvious"  # obvious | subtle | legitimate
    #: Reserved for the held-out "unseen pattern" test slice.
    holdout: bool = False
    notes: str = ""

    @property
    def is_scam(self) -> bool:
        return self.classification == Classification.SCAM.value


C, V = Speaker.CALLER, Speaker.VICTIM


# ---------------------------------------------------------------------------
# Scam families
# ---------------------------------------------------------------------------

SCAM_FAMILIES: tuple[Family, ...] = (
    Family(
        key="bank_security_fake_loan_otp",
        label="Bank security → invented loan → OTP theft",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.BANK_IMPERSONATION.value, ST.FAKE_LOAN.value, ST.OTP_THEFT.value),
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, это служба безопасности Halyk Bank, меня зовут Асель Нурланова.",
                 "Сәлеметсіз бе, бұл Halyk Bank қауіпсіздік қызметі, менің атым Әсел Нұрланова.",
                 "Сәлеметсіз бе, это служба безопасности Halyk Bank, менің атым Әсел."),
            Turn(V, SG.INTRODUCTION.value, "Да, слушаю вас.", "Иә, тыңдап тұрмын."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "На ваше имя в другом городе оформляется кредит на 2 миллиона 800 тысяч тенге.",
                 "Сіздің атыңызға басқа қалада 2 миллион 800 мың теңгеге несие рәсімделіп жатыр.",
                 "Сіздің атыңызға кредит оформляется, 2 миллиона 800 тысяч тенге.",
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value),)),
            Turn(V, SG.PROBLEM_CREATION.value,
                 "Но я ничего не оформлял. Это точно?", "Мен ондай өтінім бермедім. Дәл солай ма?"),
            Turn(C, SG.FEAR_ESCALATION.value,
                 "Именно поэтому нужно срочно остановить операцию, иначе деньги спишут в течение десяти минут.",
                 "Сондықтан операцияны дереу тоқтату керек, әйтпесе он минут ішінде ақша есептен шығады.",
                 None,
                 ((EC.URGENCY.value, SV.MEDIUM.value), (EC.THREAT.value, SV.HIGH.value))),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Не отключайтесь и никому не говорите об этом разговоре, идёт проверка.",
                 "Байланысты үзбеңіз және бұл әңгіме туралы ешкімге айтпаңыз, тексеру жүріп жатыр.",
                 None,
                 ((EC.ISOLATION.value, SV.HIGH.value), (EC.SECRECY.value, SV.HIGH.value))),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Чтобы отменить заявку, система сейчас пришлёт вам подтверждение.",
                 "Өтінімді болдырмау үшін жүйе сізге қазір растау жібереді.",
                 "Өтінімді отменить ету үшін жүйе сізге растау жібереді.",
                 ((EC.FALSE_SOLUTION.value, SV.MEDIUM.value),)),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Вам пришёл код из СМС? Продиктуйте его мне, я введу отмену.",
                 "Сізге СМС-код келді ме? Оны маған айтыңыз, мен болдырмауды енгізем.",
                 "Сізге СМС келді ме? Кодты айтып жіберіңіз, я сейчас отмену введу.",
                 ((EC.OTP_REQUEST.value, SV.CRITICAL.value),)),
            Turn(V, SG.CREDENTIAL_EXTRACTION.value,
                 "Хорошо, сейчас посмотрю.", "Жарайды, қазір қараймын."),
        ),
    ),
    Family(
        key="account_compromise_safe_account",
        label="Compromised account → transfer to a 'safe account'",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(
            ST.BANK_IMPERSONATION.value, ST.ACCOUNT_COMPROMISE.value,
            ST.SAFE_ACCOUNT_SCAM.value, ST.MONEY_TRANSFER_SCAM.value,
        ),
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Добрый день, вас беспокоит отдел финансового мониторинга Национального Банка.",
                 "Қайырлы күн, сізді Ұлттық Банктің қаржы мониторингі бөлімі алаңдатып отыр.",
                 "Қайырлы күн, вас беспокоит Ұлттық Банктің финансовый мониторинг бөлімі."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "К вашему счёту получили доступ третьи лица, зафиксирована попытка списания.",
                 "Сіздің шотыңызға үшінші тұлғалар қол жеткізген, есептен шығару әрекеті тіркелген.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value),)),
            Turn(V, SG.PROBLEM_CREATION.value, "Как это возможно?", "Бұл қалай болды?"),
            Turn(C, SG.FEAR_ESCALATION.value,
                 "Ваши деньги в опасности, счёт будет заблокирован вместе со средствами.",
                 "Ақшаңыз қауіпте, шот қаражатпен бірге бұғатталады.",
                 None,
                 ((EC.FEAR.value, SV.MEDIUM.value), (EC.THREAT.value, SV.HIGH.value))),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Единственный способ сохранить средства — перевести их на защищённый резервный счёт банка.",
                 "Қаражатты сақтаудың жалғыз жолы — оны банктің қорғалған сақтық шотына аудару.",
                 "Ақшаны сақтау үшін жалғыз жол — оны безопасный счётқа аудару.",
                 ((EC.FALSE_SOLUTION.value, SV.MEDIUM.value), (EC.SAFE_ACCOUNT.value, SV.CRITICAL.value))),
            Turn(C, SG.MONEY_TRANSFER.value,
                 "Подойдите к банкомату и внесите всю сумму по номеру, который я продиктую. Срочно.",
                 "Банкоматқа барыңыз және мен айтатын нөмір бойынша барлық соманы салыңыз. Дереу.",
                 None,
                 ((EC.MONEY_TRANSFER.value, SV.CRITICAL.value), (EC.URGENCY.value, SV.MEDIUM.value))),
            Turn(C, SG.MONEY_TRANSFER.value,
                 "Сотрудникам отделения не сообщайте цель операции, это тайна проверки.",
                 "Бөлімше қызметкерлеріне операцияның мақсатын айтпаңыз, бұл тексеру құпиясы.",
                 None,
                 ((EC.SECRECY.value, SV.HIGH.value), (EC.ISOLATION.value, SV.HIGH.value))),
            Turn(V, SG.MONEY_TRANSFER.value, "Хорошо, сейчас поеду.", "Жарайды, қазір барамын."),
        ),
    ),
    Family(
        key="police_investigator_money_mule",
        label="Police / investigator → criminal case → money transfer",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(
            ST.POLICE_IMPERSONATION.value, ST.INVESTIGATOR_IMPERSONATION.value,
            ST.MONEY_TRANSFER_SCAM.value,
        ),
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, это старший следователь департамента полиции Алматы, капитан Ержанов.",
                 "Сәлеметсіз бе, бұл Алматы полиция департаментінің аға тергеушісі, капитан Ержанов.",
                 "Сәлеметсіз бе, это старший следователь полиция департаментінің, капитан Ержанов."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "На ваши реквизиты возбуждено уголовное дело о финансировании мошеннической схемы.",
                 "Сіздің деректемелеріңізге алаяқтық схеманы қаржыландыру туралы қылмыстық іс қозғалды.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value),)),
            Turn(V, SG.PROBLEM_CREATION.value,
                 "Я не понимаю, я ничего не делал.", "Түсінмедім, мен ештеңе жасаған жоқпын."),
            Turn(C, SG.FEAR_ESCALATION.value,
                 "Если не подтвердите законность средств, вас привлекут к уголовной ответственности.",
                 "Егер қаражаттың заңдылығын растамасаңыз, сізді қылмыстық жауапкершілікке тартады.",
                 None,
                 ((EC.THREAT.value, SV.HIGH.value),)),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Разговор проходит в рамках тайны следствия, никому не сообщайте, включая родственников.",
                 "Әңгіме тергеу құпиясы шеңберінде өтеді, туыстарыңызды қоса, ешкімге айтпаңыз.",
                 None,
                 ((EC.SECRECY.value, SV.HIGH.value),)),
            Turn(C, SG.MONEY_TRANSFER.value,
                 "Для проверки происхождения денег переведите их на транзитный счёт следствия, я дам реквизиты.",
                 "Ақшаның шығу тегін тексеру үшін оны тергеудің транзиттік шотына аударыңыз, деректемелерді берем.",
                 "Ақшаны тексеру үшін оны транзитный счётқа аударыңыз, я реквизиты берем.",
                 ((EC.MONEY_TRANSFER.value, SV.CRITICAL.value), (EC.SAFE_ACCOUNT.value, SV.CRITICAL.value))),
            Turn(C, SG.MONEY_TRANSFER.value,
                 "Не кладите трубку и не звоните в банк, иначе это будет расценено как сокрытие.",
                 "Телефонды тастамаңыз және банкке қоңырау соқпаңыз, әйтпесе бұл жасыру деп саналады.",
                 None,
                 ((EC.ISOLATION.value, SV.HIGH.value), (EC.THREAT.value, SV.HIGH.value))),
        ),
    ),
    Family(
        key="fake_support_remote_access",
        label="Fake technical support → remote access",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.BANK_IMPERSONATION.value, ST.REMOTE_ACCESS_SCAM.value),
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, техподдержка Kaspi, специалист Данияр.",
                 "Сәлеметсіз бе, Kaspi техқолдау қызметі, маман Данияр.",
                 "Сәлеметсіз бе, Kaspi техподдержка, маман Данияр."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "В вашем приложении обнаружена подозрительная операция и сбой в системе безопасности.",
                 "Сіздің қолданбада күдікті операция және қауіпсіздік жүйесінде іркіліс анықталды.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value),)),
            Turn(V, SG.PROBLEM_CREATION.value, "И что теперь делать?", "Ал енді не істеу керек?"),
            Turn(C, SG.REMOTE_ACCESS.value,
                 "Установите приложение AnyDesk по ссылке, которую я отправлю, я подключусь и всё исправлю.",
                 "Мен жіберетін сілтеме бойынша AnyDesk қолданбасын орнатыңыз, мен қосылып бәрін түзетем.",
                 "Мен жіберетін ссылка бойынша AnyDesk орнатыңыз, я подключусь и всё исправлю.",
                 ((EC.APP_INSTALLATION.value, SV.HIGH.value), (EC.REMOTE_ACCESS.value, SV.CRITICAL.value))),
            Turn(C, SG.REMOTE_ACCESS.value,
                 "Включите демонстрацию экрана и откройте банковское приложение, я проверю баланс.",
                 "Экранды көрсетуді қосыңыз және банк қолданбасын ашыңыз, мен балансты тексерем.",
                 None,
                 ((EC.SCREEN_SHARING.value, SV.HIGH.value),)),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Сейчас придёт push-код, назовите его, иначе доступ не восстановится.",
                 "Қазір push-код келеді, оны айтыңыз, әйтпесе рұқсат қалпына келмейді.",
                 None,
                 ((EC.OTP_REQUEST.value, SV.CRITICAL.value), (EC.URGENCY.value, SV.MEDIUM.value))),
        ),
    ),
    Family(
        key="marketplace_buyer_card_theft",
        label="Marketplace 'buyer' → card data theft",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.MARKETPLACE_SCAM.value, ST.CARD_DATA_THEFT.value),
        difficulty="subtle",
        turns=(
            Turn(C, SG.INTRODUCTION.value,
                 "Здравствуйте, я по объявлению на OLX, беру ваш диван, всё устраивает.",
                 "Сәлеметсіз бе, мен OLX-тегі хабарландыру бойынша, диванды аламын, бәрі жарайды.",
                 "Сәлеметсіз бе, я по объявлению на OLX, диванды аламын."),
            Turn(V, SG.INTRODUCTION.value, "Отлично, когда заберёте?", "Жақсы, қашан алып кетесіз?"),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Я оплачу через доставку Kaspi прямо сейчас, оператор уже оформил перевод.",
                 "Мен қазір Kaspi жеткізу арқылы төлеймін, оператор аударымды рәсімдеп қойды.",
                 None),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Чтобы деньги зашли, продиктуйте номер карты полностью, срок действия и три цифры на обороте.",
                 "Ақша түсуі үшін карта нөмірін толық, жарамдылық мерзімін және артындағы үш санды айтыңыз.",
                 "Ақша түсу үшін номер карты толық, срок действия және артындағы үш санды айтыңыз.",
                 ((EC.CARD_REQUEST.value, SV.CRITICAL.value),)),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "А зачем CVV для получения денег?", "Ақша алу үшін CVV неге керек?"),
            Turn(C, SG.FEAR_ESCALATION.value,
                 "Так работает система, иначе перевод отменится и вы потеряете покупателя. Быстрее, у меня мало времени.",
                 "Жүйе солай жұмыс істейді, әйтпесе аударым жойылады және сатып алушыдан айырыласыз. Тезірек, уақытым жоқ.",
                 None,
                 ((EC.URGENCY.value, SV.MEDIUM.value), (EC.FEAR.value, SV.MEDIUM.value))),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "И код из СМС тоже назовите, он подтверждает зачисление.",
                 "СМС-тегі кодты да айтыңыз, ол есепке алуды растайды.",
                 None,
                 ((EC.OTP_REQUEST.value, SV.CRITICAL.value),)),
        ),
    ),
    Family(
        key="delivery_courier_otp",
        label="Courier confirming delivery → OTP theft",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.DELIVERY_SCAM.value, ST.OTP_THEFT.value),
        difficulty="subtle",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, служба доставки, курьер Санжар, у меня ваша посылка.",
                 "Сәлеметсіз бе, жеткізу қызметі, курьер Санжар, сәлемдемеңіз менде.",
                 "Сәлеметсіз бе, служба доставки, курьер Санжар, сәлемдемеңіз менде."),
            Turn(V, SG.INTRODUCTION.value,
                 "Я ничего не заказывал вроде.", "Мен тапсырыс бермеген сияқтымын."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "Заказ оформлен на ваш номер, если не подтвердить, посылка уйдёт обратно и спишется оплата.",
                 "Тапсырыс сіздің нөміріңізге рәсімделген, растамасаңыз, сәлемдеме қайтады және төлем есептен шығады.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.MEDIUM.value), (EC.THREAT.value, SV.HIGH.value))),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Сейчас вам придёт код подтверждения, назовите его мне, и я закрою доставку.",
                 "Қазір сізге растау коды келеді, оны маған айтыңыз, мен жеткізуді жабамын.",
                 "Қазір сізге код подтверждения келеді, оны айтыңыз, я закрою доставку.",
                 ((EC.OTP_REQUEST.value, SV.CRITICAL.value),)),
            Turn(V, SG.CREDENTIAL_EXTRACTION.value, "Секунду...", "Бір минут..."),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Только быстрее, у меня ещё двадцать адресов.",
                 "Тезірек болыңыз, менде әлі жиырма мекенжай бар.",
                 None,
                 ((EC.URGENCY.value, SV.MEDIUM.value),)),
        ),
    ),
    Family(
        key="egov_benefit_credential",
        label="eGov 'benefit payment' → account credential theft",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.GOVERNMENT_IMPERSONATION.value, ST.CREDENTIAL_THEFT.value),
        difficulty="subtle",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, центр государственных услуг, вопрос по вашей выплате через eGov.",
                 "Сәлеметсіз бе, мемлекеттік қызметтер орталығы, eGov арқылы төлеміңіз бойынша сұрақ.",
                 "Сәлеметсіз бе, гос услуги орталығы, eGov арқылы төлеміңіз бойынша сұрақ."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "Вам положен возврат переплаты, но заявка зависла из-за несовпадения данных.",
                 "Сізге артық төлемді қайтару тиесілі, бірақ деректер сәйкес келмегендіктен өтінім тұрып қалды.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.MEDIUM.value),)),
            Turn(V, SG.PROBLEM_CREATION.value,
                 "И что нужно сделать?", "Не істеу керек?"),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Назовите ваш ИИН и дату рождения для сверки.",
                 "Салыстыру үшін ЖСН мен туған күніңізді айтыңыз.",
                 None,
                 ((EC.PERSONAL_DATA_REQUEST.value, SV.MEDIUM.value),)),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Теперь пароль от личного кабинета eGov, я обновлю заявку от вашего имени.",
                 "Енді eGov жеке кабинетінің құпия сөзін айтыңыз, өтінімді сіздің атыңыздан жаңартам.",
                 "Енді eGov личный кабинет паролін айтыңыз, я обновлю заявку.",
                 ((EC.PASSWORD_REQUEST.value, SV.CRITICAL.value),)),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Выплата действует до конца дня, потом заявка закроется.",
                 "Төлем күн соңына дейін жарамды, содан кейін өтінім жабылады.",
                 None,
                 ((EC.URGENCY.value, SV.MEDIUM.value),)),
        ),
    ),
    Family(
        key="investment_crypto_profit",
        label="Investment platform → crypto deposit",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.INVESTMENT_SCAM.value, ST.CRYPTO_SCAM.value),
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Добрый день, я персональный брокер инвестиционной платформы, меня зовут Алексей.",
                 "Қайырлы күн, мен инвестициялық платформаның жеке брокерімін, менің атым Алексей.",
                 "Қайырлы күн, мен инвестиционный платформаның брокерімін, менің атым Алексей."),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Наши клиенты получают гарантированный доход двенадцать процентов в месяц, риска нет.",
                 "Клиенттеріміз айына он екі пайыз кепілдендірілген табыс алады, тәуекел жоқ.",
                 None),
            Turn(V, SG.INTRODUCTION.value,
                 "Звучит слишком хорошо. А как это работает?",
                 "Тым жақсы естіледі. Бұл қалай жұмыс істейді?"),
            Turn(C, SG.PAYMENT.value,
                 "Пополните торговый счёт в USDT через обменник, минимальный депозит пятьсот долларов.",
                 "Айырбастау пункті арқылы сауда шотын USDT-мен толтырыңыз, ең аз депозит бес жүз доллар.",
                 "Обменник арқылы торговый счётты USDT-мен толтырыңыз, минимальный депозит бес жүз доллар.",
                 ((EC.CRYPTO_TRANSFER.value, SV.HIGH.value), (EC.MONEY_TRANSFER.value, SV.HIGH.value))),
            Turn(C, SG.PAYMENT.value,
                 "Акция заканчивается сегодня, потом вход только от пяти тысяч.",
                 "Акция бүгін бітеді, кейін кіру бес мыңнан басталады.",
                 None,
                 ((EC.URGENCY.value, SV.MEDIUM.value),)),
            Turn(C, SG.REMOTE_ACCESS.value,
                 "Установите наше приложение по ссылке, я помогу настроить счёт удалённо.",
                 "Сілтеме бойынша қолданбамызды орнатыңыз, мен шотты қашықтан реттеуге көмектесем.",
                 None,
                 ((EC.APP_INSTALLATION.value, SV.HIGH.value), (EC.REMOTE_ACCESS.value, SV.CRITICAL.value))),
        ),
    ),
    Family(
        key="job_offer_advance_fee",
        label="Job offer → advance fee transfer",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.JOB_SCAM.value, ST.MONEY_TRANSFER_SCAM.value),
        difficulty="subtle",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, я из отдела подбора персонала, по вашей вакансии на удалённую работу.",
                 "Сәлеметсіз бе, мен персонал таңдау бөлімінен, қашықтан жұмыс вакансиясы бойынша.",
                 "Сәлеметсіз бе, мен HR отделынан, қашықтан жұмыс вакансиясы бойынша."),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Оплата от четырёхсот тысяч тенге в месяц, график свободный, обучение два дня.",
                 "Айына төрт жүз мың теңгеден төлем, кестесі еркін, оқыту екі күн.",
                 None),
            Turn(V, SG.INTRODUCTION.value, "Интересно. Что нужно от меня?", "Қызық. Менен не керек?"),
            Turn(C, SG.PAYMENT.value,
                 "Для оформления внесите депозит двадцать тысяч тенге за пакет документов, потом вернём с первой зарплатой.",
                 "Рәсімдеу үшін құжаттар пакетіне жиырма мың теңге депозит салыңыз, кейін алғашқы жалақымен қайтарамыз.",
                 "Рәсімдеу үшін жиырма мың теңге депозит салыңыз, потом с первой зарплатой қайтарамыз.",
                 ((EC.MONEY_TRANSFER.value, SV.HIGH.value),)),
            Turn(C, SG.PAYMENT.value,
                 "Место держим до вечера, желающих много.",
                 "Орынды кешке дейін ұстаймыз, қалаушылар көп.",
                 None,
                 ((EC.URGENCY.value, SV.MEDIUM.value),)),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "И пришлите фото удостоверения личности с двух сторон.",
                 "Жеке куәліктің екі жағының фотосын жіберіңіз.",
                 None,
                 ((EC.PERSONAL_DATA_REQUEST.value, SV.MEDIUM.value),)),
        ),
    ),
    Family(
        key="fake_refund_card",
        label="Fake refund → card credentials",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.FAKE_REFUND.value, ST.CARD_DATA_THEFT.value),
        difficulty="subtle",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, интернет-магазин, отдел возвратов, по вашему отменённому заказу.",
                 "Сәлеметсіз бе, интернет-дүкен, қайтару бөлімі, жойылған тапсырысыңыз бойынша.",
                 "Сәлеметсіз бе, интернет-магазин, отдел возвратов, жойылған тапсырысыңыз бойынша."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "Вам положена компенсация двадцать девять тысяч тенге, но реквизиты не подтянулись.",
                 "Сізге жиырма тоғыз мың теңге өтемақы тиесілі, бірақ деректемелер тартылмады.",
                 None,
                 ((EC.PROBLEM_LEGAL.value if False else EC.FALSE_PROBLEM.value, SV.MEDIUM.value),)),
            Turn(V, SG.PROBLEM_CREATION.value, "А какой заказ?", "Қандай тапсырыс?"),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Заказ от прошлой недели. Продиктуйте данные карты для зачисления возврата.",
                 "Өткен аптадағы тапсырыс. Қайтаруды есепке алу үшін карта деректерін айтыңыз.",
                 "Өткен аптадағы заказ. Возврат үшін данные карты айтыңыз.",
                 ((EC.CARD_REQUEST.value, SV.CRITICAL.value),)),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "И код, который придёт в сообщении, он подтвердит возврат.",
                 "Хабарламада келетін кодты да айтыңыз, ол қайтаруды растайды.",
                 None,
                 ((EC.OTP_REQUEST.value, SV.CRITICAL.value),)),
        ),
    ),
    Family(
        key="telco_sim_swap_otp",
        label="Mobile operator 'SIM renewal' → OTP interception",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.OTHER_SOCIAL_ENGINEERING.value, ST.OTP_THEFT.value),
        difficulty="subtle",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, оператор Beeline, отдел безопасности номеров.",
                 "Сәлеметсіз бе, Beeline операторы, нөмірлер қауіпсіздігі бөлімі.",
                 "Сәлеметсіз бе, Beeline оператор, нөмірлер қауіпсіздік бөлімі."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "На вашу сим-карту поступила заявка на перевыпуск от третьего лица.",
                 "Сіздің сим-картаңызға үшінші тұлғадан қайта шығаруға өтінім түсті.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value),)),
            Turn(V, SG.PROBLEM_CREATION.value, "Я не подавал такую заявку!", "Мен ондай өтінім бермедім!"),
            Turn(C, SG.FEAR_ESCALATION.value,
                 "Если не отменить сейчас, номер уйдёт мошенникам вместе с доступом к банку.",
                 "Қазір болдырмасаңыз, нөмір банкке рұқсатпен бірге алаяқтарға кетеді.",
                 None,
                 ((EC.THREAT.value, SV.HIGH.value), (EC.FEAR.value, SV.MEDIUM.value))),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Диктуйте код из СМС, которое сейчас придёт, это отмена перевыпуска.",
                 "Қазір келетін СМС-тегі кодты айтыңыз, бұл қайта шығаруды болдырмау.",
                 "Қазір келетін СМС кодты айтыңыз, это отмена перевыпуска.",
                 ((EC.OTP_REQUEST.value, SV.CRITICAL.value),)),
        ),
    ),
    Family(
        key="screen_share_banking",
        label="'Compliance check' → screen sharing during a banking session",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.BANK_IMPERSONATION.value, ST.REMOTE_ACCESS_SCAM.value),
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, антифрод-мониторинг ForteBank, специалист Гульнара.",
                 "Сәлеметсіз бе, ForteBank антифрод мониторингі, маман Гүлнара.",
                 "Сәлеметсіз бе, ForteBank антифрод мониторинг, маман Гүлнара."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "По вашей карте прошла несанкционированная операция в другой стране.",
                 "Сіздің картаңыз бойынша басқа елде рұқсат етілмеген операция өтті.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value),)),
            Turn(V, SG.PROBLEM_CREATION.value, "Я за границей не был.", "Мен шетелде болмадым."),
            Turn(C, SG.REMOTE_ACCESS.value,
                 "Поделитесь экраном в мессенджере и откройте приложение банка, я покажу, где отменить.",
                 "Мессенджерде экранды бөлісіңіз және банк қолданбасын ашыңыз, қайдан болдырмауды көрсетем.",
                 "Мессенджерде экранды бөлісіңіз, банк қолданбасын ашыңыз, я покажу где отменить.",
                 ((EC.SCREEN_SHARING.value, SV.HIGH.value),)),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "Введите код из push при мне, только не закрывайте демонстрацию.",
                 "Push-тегі кодты мен көріп отырғанда енгізіңіз, тек көрсетуді жаппаңыз.",
                 None,
                 ((EC.OTP_REQUEST.value, SV.CRITICAL.value), (EC.ISOLATION.value, SV.HIGH.value))),
        ),
    ),
    # -- reserved for the "unseen pattern" test slice -------------------
    Family(
        key="bank_app_update_apk",
        label="'Mandatory app update' → malicious APK (held out)",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(ST.BANK_IMPERSONATION.value, ST.REMOTE_ACCESS_SCAM.value),
        holdout=True,
        difficulty="subtle",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Добрый день, отдел безопасности Jusan Bank, по обязательному обновлению приложения.",
                 "Қайырлы күн, Jusan Bank қауіпсіздік бөлімі, қолданбаны міндетті жаңарту бойынша.",
                 "Қайырлы күн, Jusan Bank қауіпсіздік бөлімі, обязательное обновление бойынша."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "Ваша версия приложения устарела, поэтому карта будет заблокирована сегодня.",
                 "Қолданба нұсқасы ескірген, сондықтан карта бүгін бұғатталады.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value), (EC.THREAT.value, SV.HIGH.value))),
            Turn(V, SG.PROBLEM_CREATION.value,
                 "У меня всё работает вроде.", "Менде бәрі жұмыс істеп тұр сияқты."),
            Turn(C, SG.REMOTE_ACCESS.value,
                 "Я отправлю ссылку, загрузите apk файл и установите поверх старого приложения.",
                 "Мен сілтеме жіберем, apk файлды жүктеп алыңыз және ескі қолданбаның үстінен орнатыңыз.",
                 "Мен ссылка жіберем, apk файлды жүктеп, ескі қолданбаның үстінен орнатыңыз.",
                 ((EC.APP_INSTALLATION.value, SV.HIGH.value),)),
            Turn(C, SG.CREDENTIAL_EXTRACTION.value,
                 "После установки введите логин и пароль, потом назовите мне код из СМС для синхронизации.",
                 "Орнатқаннан кейін логин мен құпия сөзді енгізіңіз, содан кейін синхрондау үшін СМС кодын айтыңыз.",
                 None,
                 ((EC.PASSWORD_REQUEST.value, SV.CRITICAL.value), (EC.OTP_REQUEST.value, SV.CRITICAL.value))),
        ),
    ),
    Family(
        key="prosecutor_secrecy_cash_deposit",
        label="Prosecutor's office → secret cash deposit (held out)",
        classification="SCAM",
        call_direction="outbound",
        scam_types=(
            ST.INVESTIGATOR_IMPERSONATION.value, ST.GOVERNMENT_IMPERSONATION.value,
            ST.SAFE_ACCOUNT_SCAM.value,
        ),
        holdout=True,
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, помощник прокурора района, дело по вашему счёту.",
                 "Сәлеметсіз бе, аудан прокурорының көмекшісі, шотыңыз бойынша іс.",
                 "Сәлеметсіз бе, прокурор көмекшісі, шотыңыз бойынша іс."),
            Turn(C, SG.PROBLEM_CREATION.value,
                 "Ваш счёт использовали для обналичивания, есть задолженность и исполнительное производство.",
                 "Шотыңыз қолма-қол айналдыру үшін пайдаланылған, берешек және орындау ісі бар.",
                 None,
                 ((EC.FALSE_PROBLEM.value, SV.HIGH.value),)),
            Turn(V, SG.PROBLEM_CREATION.value, "Это какая-то ошибка.", "Бұл қандай да бір қателік."),
            Turn(C, SG.FEAR_ESCALATION.value,
                 "Без подтверждения средств будет арест счетов и повестка в суд.",
                 "Қаражат расталмаса, шоттарға тыйым салынады және сотқа шақыру келеді.",
                 None,
                 ((EC.THREAT.value, SV.HIGH.value),)),
            Turn(C, SG.MONEY_TRANSFER.value,
                 "Снимите наличные и внесите на специальный счёт через терминал, реквизиты сейчас пришлю.",
                 "Қолма-қол ақшаны алып, терминал арқылы арнайы шотқа салыңыз, деректемелерді қазір жіберем.",
                 "Наличные алып, терминал арқылы арнайы шотқа салыңыз, реквизиты қазір жіберем.",
                 ((EC.MONEY_TRANSFER.value, SV.CRITICAL.value), (EC.SAFE_ACCOUNT.value, SV.CRITICAL.value))),
            Turn(C, SG.MONEY_TRANSFER.value,
                 "Разговор конфиденциальный, никому не сообщайте, иначе будет статья за разглашение.",
                 "Әңгіме құпия, ешкімге айтпаңыз, әйтпесе жария ету үшін бап болады.",
                 None,
                 ((EC.SECRECY.value, SV.HIGH.value), (EC.ISOLATION.value, SV.HIGH.value))),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Legitimate families (hard negatives — spec section 14)
# ---------------------------------------------------------------------------

LEGIT_FAMILIES: tuple[Family, ...] = (
    Family(
        key="inbound_failed_payment",
        label="Customer calls the bank about a declined payment",
        classification="SAFE",
        call_direction="inbound",
        difficulty="legitimate",
        turns=(
            Turn(V, SG.INTRODUCTION.value,
                 "Здравствуйте, я хотел узнать, почему не прошла оплата картой в магазине.",
                 "Сәлеметсіз бе, дүкенде картамен төлем неге өтпегенін білгім келеді.",
                 "Сәлеметсіз бе, я хотел узнать, дүкенде төлем неге өтпеді."),
            Turn(C, SG.INTRODUCTION.value,
                 "Здравствуйте, это Halyk Bank, оператор Мадина. Назовите последние четыре цифры карты для идентификации.",
                 "Сәлеметсіз бе, бұл Halyk Bank, оператор Мадина. Сәйкестендіру үшін картаның соңғы төрт санын айтыңыз.",
                 None,
                 ((EC.PERSONAL_DATA_REQUEST.value, SV.LOW.value),)),
            Turn(V, SG.INFORMATION_EXTRACTION.value, "Четыре, восемь, два, один.", "Төрт, сегіз, екі, бір."),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Вижу отклонение по лимиту на интернет-операции. Лимит можно изменить в приложении сами.",
                 "Интернет-операциялар лимиті бойынша қабылданбауды көріп отырмын. Лимитті қолданбада өзіңіз өзгертуге болады.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(V, SG.EXIT.value, "Понятно, спасибо большое.", "Түсінікті, көп рахмет."),
            Turn(C, SG.EXIT.value,
                 "Пожалуйста. Код из СМС мы никогда не спрашиваем, никому его не сообщайте.",
                 "Оқасы жоқ. СМС-кодты біз ешқашан сұрамаймыз, оны ешкімге айтпаңыз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
        ),
    ),
    Family(
        key="inbound_card_freeze",
        label="Customer asks how to freeze a card",
        classification="SAFE",
        call_direction="inbound",
        difficulty="legitimate",
        turns=(
            Turn(V, SG.INTRODUCTION.value,
                 "Подскажите пожалуйста, как заблокировать карту, я потерял её.",
                 "Айтып жіберіңізші, картаны қалай бұғаттауға болады, мен оны жоғалтып алдым.",
                 "Айтып жіберіңізші, картаны қалай заблокировать ету керек, я её потерял."),
            Turn(C, SG.INTRODUCTION.value,
                 "Здравствуйте, конечно. Карту можно заблокировать в приложении в разделе безопасность или здесь по телефону.",
                 "Сәлеметсіз бе, әрине. Картаны қолданбада қауіпсіздік бөлімінде немесе осы жерде телефонмен бұғаттауға болады.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "Давайте по телефону, приложение не открывается.",
                 "Телефонмен жасайық, қолданба ашылмайды."),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Хорошо. Подтвердите кодовое слово по счёту, пожалуйста.",
                 "Жарайды. Шот бойынша кодтық сөзді растаңыз.",
                 None,
                 ((EC.PERSONAL_DATA_REQUEST.value, SV.LOW.value),)),
            Turn(V, SG.INFORMATION_EXTRACTION.value, "Тюльпан.", "Қызғалдақ."),
            Turn(C, SG.EXIT.value,
                 "Карта заблокирована. Перевыпуск оформите в отделении или в приложении.",
                 "Карта бұғатталды. Қайта шығаруды бөлімшеде немесе қолданбада рәсімдейсіз."),
        ),
    ),
    Family(
        key="bank_security_education",
        label="Bank calls to warn the customer never to share an OTP",
        classification="SAFE",
        call_direction="outbound",
        difficulty="legitimate",
        notes="Contains OTP, bank, and security vocabulary with no fraudulent request.",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, это Kaspi, информационная рассылка о безопасности, звонок без запросов.",
                 "Сәлеметсіз бе, бұл Kaspi, қауіпсіздік туралы ақпараттық хабарлама, сұраныссыз қоңырау.",
                 "Сәлеметсіз бе, бұл Kaspi, безопасность туралы информационная рассылка."),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Напоминаем: сотрудник банка никогда не спрашивает код из СМС, пароль или CVV.",
                 "Еске саламыз: банк қызметкері ешқашан СМС-кодты, құпия сөзді немесе CVV-ді сұрамайды.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(V, SG.INTRODUCTION.value, "Да, я знаю, спасибо.", "Иә, білемін, рахмет."),
            Turn(C, SG.TRUST_BUILDING.value,
                 "Если вам звонят и просят перевести деньги на безопасный счёт — это мошенники, положите трубку.",
                 "Егер сізге қоңырау соғып, ақшаны қауіпсіз шотқа аударуды сұраса — бұл алаяқтар, телефонды қойыңыз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(C, SG.EXIT.value,
                 "Проверить любую информацию можно по номеру на обратной стороне карты. Хорошего дня.",
                 "Кез келген ақпаратты картаның артқы жағындағы нөмір бойынша тексеруге болады. Күніңіз жақсы өтсін.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
        ),
    ),
    Family(
        key="outbound_loan_followup",
        label="Bank follows up on a loan application the customer submitted",
        classification="SAFE",
        call_direction="outbound",
        difficulty="legitimate",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, я звоню из ForteBank по поводу вашей заявки на кредит от вторника.",
                 "Сәлеметсіз бе, сейсенбіде берген несие өтініміңіз бойынша ForteBank-тен қоңырау соғып отырмын.",
                 "Сәлеметсіз бе, я звоню из ForteBank, сейсенбідегі кредит өтініміңіз бойынша."),
            Turn(V, SG.INTRODUCTION.value, "Да, я подавал заявку.", "Иә, мен өтінім бердім."),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Заявка предварительно одобрена на два миллиона тенге под двадцать один процент.",
                 "Өтінім жиырма бір пайызбен екі миллион теңгеге алдын ала мақұлданды.",
                 None),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "А как подписать договор?", "Шартқа қалай қол қоюға болады?"),
            Turn(C, SG.EXIT.value,
                 "Подписать можно в мобильном приложении или в любом отделении с удостоверением личности.",
                 "Мобильді қолданбада немесе жеке куәлікпен кез келген бөлімшеде қол қоюға болады.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(C, SG.EXIT.value,
                 "Никаких кодов по телефону мы не запрашиваем.",
                 "Телефон арқылы ешқандай код сұрамаймыз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
        ),
    ),
    Family(
        key="inbound_report_suspicious_transaction",
        label="Customer reports a suspicious transaction",
        classification="SAFE",
        call_direction="inbound",
        difficulty="legitimate",
        notes="Customer-side use of fraud vocabulary (мошенники, списание) with no attack.",
        turns=(
            Turn(V, SG.INTRODUCTION.value,
                 "Здравствуйте, у меня с карты списали двадцать тысяч, я эту операцию не делал.",
                 "Сәлеметсіз бе, картамнан жиырма мың теңге шығып кеткен, мен ол операцияны жасаған жоқпын.",
                 "Сәлеметсіз бе, картамнан жиырма мың списали, я эту операцию не делал."),
            Turn(C, SG.INTRODUCTION.value,
                 "Здравствуйте, зафиксирую обращение. Карту сейчас заблокируем, чтобы остановить списания.",
                 "Сәлеметсіз бе, өтінішті тіркеймін. Шығындарды тоқтату үшін картаны қазір бұғаттаймыз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "Да, заблокируйте пожалуйста. Это мошенники какие-то.",
                 "Иә, бұғаттаңыз. Бұл қандай да бір алаяқтар."),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Карта заблокирована. Оформим претензию, срок рассмотрения до тридцати дней.",
                 "Карта бұғатталды. Талап-арыз рәсімдейміз, қарау мерзімі отыз күнге дейін."),
            Turn(V, SG.EXIT.value, "Хорошо, спасибо.", "Жарайды, рахмет."),
        ),
    ),
    Family(
        key="police_explains_reporting",
        label="Police officer explains how to report a scam",
        classification="SAFE",
        call_direction="inbound",
        difficulty="legitimate",
        notes="Police + criminal case + money vocabulary, protective intent.",
        turns=(
            Turn(V, SG.INTRODUCTION.value,
                 "Здравствуйте, мне звонили мошенники, представлялись банком. Что делать?",
                 "Сәлеметсіз бе, маған алаяқтар қоңырау соқты, банк деп таныстырды. Не істеу керек?",
                 "Сәлеметсіз бе, маған мошенники қоңырау соқты, банк деп таныстырды. Что делать?"),
            Turn(C, SG.INTRODUCTION.value,
                 "Это участковый инспектор полиции. Если денег не перевели, ущерба нет, но заявление подать стоит.",
                 "Бұл полиция участкелік инспекторы. Ақша аудармаған болсаңыз, зиян жоқ, бірақ арыз беру керек.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "А куда обращаться?", "Қайда хабарласу керек?"),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Заявление можно подать в отделении полиции или через eGov, приложите скриншоты и номер звонившего.",
                 "Арызды полиция бөлімшесінде немесе eGov арқылы беруге болады, скриншоттар мен қоңырау шалған нөмірді қосыңыз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(C, SG.EXIT.value,
                 "И запомните: никогда не сообщайте код из СМС и не переводите деньги по просьбе по телефону.",
                 "Есте сақтаңыз: СМС-кодты ешқашан айтпаңыз және телефон арқылы өтініш бойынша ақша аудармаңыз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
        ),
    ),
    Family(
        key="genuine_courier_address",
        label="Real courier confirming an address",
        classification="SAFE",
        call_direction="outbound",
        difficulty="legitimate",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, курьер службы доставки, у меня посылка на ваш адрес.",
                 "Сәлеметсіз бе, жеткізу қызметінің курьері, менде сіздің мекенжайыңызға сәлемдеме бар.",
                 "Сәлеметсіз бе, курьер службы доставки, сізге сәлемдеме бар."),
            Turn(V, SG.INTRODUCTION.value,
                 "Да, я заказывал наушники.", "Иә, мен құлаққап тапсырыс бердім."),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Подъезд второй, домофон работает? Буду через двадцать минут.",
                 "Кіреберіс екінші ме, домофон жұмыс істейді ме? Жиырма минуттан кейін боламын."),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "Да, работает, поднимайтесь.", "Иә, жұмыс істейді, көтеріліңіз."),
            Turn(C, SG.EXIT.value,
                 "Оплата при получении картой или наличными, код подтверждения назовёте вы мне из своего приложения.",
                 "Төлем алу кезінде картамен немесе қолма-қол, растау кодын өз қолданбаңыздан өзіңіз көрсетесіз."),
        ),
    ),
    Family(
        key="telco_tariff_offer",
        label="Mobile operator offers a tariff (marketing, no credentials)",
        classification="SAFE",
        call_direction="outbound",
        difficulty="legitimate",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, это оператор Activ, предложение по вашему тарифу.",
                 "Сәлеметсіз бе, бұл Activ операторы, тарифіңіз бойынша ұсыныс.",
                 "Сәлеметсіз бе, бұл Activ оператор, тарифіңіз бойынша предложение."),
            Turn(C, SG.TRUST_BUILDING.value,
                 "За ту же цену можно получить больше интернета, подключение через приложение или USSD.",
                 "Сол баға үшін көбірек интернет алуға болады, қосылу қолданба немесе USSD арқылы."),
            Turn(V, SG.INTRODUCTION.value,
                 "А это бесплатно?", "Бұл тегін бе?"),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Подключение бесплатное, стоимость тарифа не меняется. Решение полностью за вами.",
                 "Қосылу тегін, тариф құны өзгермейді. Шешім толықтай сізде."),
            Turn(V, SG.EXIT.value, "Хорошо, я подумаю.", "Жарайды, ойланамын."),
        ),
    ),
    Family(
        key="bank_verify_transaction_no_otp",
        label="Bank verifies a real transaction and refuses to take an OTP",
        classification="SAFE",
        call_direction="outbound",
        difficulty="legitimate",
        notes="Hardest negative: outbound bank call about a suspicious operation, no harmful request.",
        turns=(
            Turn(C, SG.IDENTITY_CLAIM.value,
                 "Здравствуйте, служба мониторинга Jusan Bank. Вы совершали покупку на сто двадцать тысяч в интернете?",
                 "Сәлеметсіз бе, Jusan Bank мониторинг қызметі. Интернетте жүз жиырма мыңға сатып алу жасадыңыз ба?",
                 "Сәлеметсіз бе, Jusan Bank мониторинг қызметі. Интернетте сто двадцать тысяч сатып алу жасадыңыз ба?"),
            Turn(V, SG.INTRODUCTION.value, "Да, это я, покупал билеты.", "Иә, бұл мен, билет сатып алдым."),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Спасибо, операцию подтверждаем, ограничения снимаем. Код из СМС мне называть не нужно.",
                 "Рахмет, операцияны растаймыз, шектеулерді алып тастаймыз. СМС-кодты маған айту қажет емес.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "А как я могу вам верить?", "Мен сізге қалай сене аламын?"),
            Turn(C, SG.EXIT.value,
                 "Правильный вопрос. Положите трубку и перезвоните по номеру на обратной стороне карты.",
                 "Дұрыс сұрақ. Телефонды қойып, картаның артқы жағындағы нөмірге қайта қоңырау соғыңыз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
        ),
    ),
    Family(
        key="investment_platform_kyc",
        label="Licensed broker completes KYC without asking for a transfer (held out)",
        classification="SAFE",
        call_direction="inbound",
        difficulty="legitimate",
        holdout=True,
        turns=(
            Turn(V, SG.INTRODUCTION.value,
                 "Здравствуйте, я открывал брокерский счёт, хочу уточнить статус.",
                 "Сәлеметсіз бе, брокерлік шот ашқанмын, статусын нақтылағым келеді.",
                 "Сәлеметсіз бе, я открывал брокерский счёт, статусын нақтылағым келеді."),
            Turn(C, SG.INTRODUCTION.value,
                 "Здравствуйте, счёт открыт, осталось пройти верификацию личности в приложении.",
                 "Сәлеметсіз бе, шот ашылды, қолданбада жеке тұлғаны растау қалды."),
            Turn(V, SG.INFORMATION_EXTRACTION.value,
                 "А деньги куда вносить?", "Ақшаны қайда салу керек?"),
            Turn(C, SG.INFORMATION_EXTRACTION.value,
                 "Только через ваш личный кабинет, реквизиты там же. По телефону мы переводы не оформляем.",
                 "Тек өзіңіздің жеке кабинетіңіз арқылы, деректемелер сол жерде. Телефон арқылы аударым рәсімдемейміз.",
                 None,
                 ((EC.PROTECTIVE_ADVICE.value, SV.LOW.value),)),
            Turn(V, SG.EXIT.value, "Понял, спасибо.", "Түсінікті, рахмет."),
        ),
    ),
)


ALL_FAMILIES: tuple[Family, ...] = SCAM_FAMILIES + LEGIT_FAMILIES
FAMILIES_BY_KEY: dict[str, Family] = {family.key: family for family in ALL_FAMILIES}


# ---------------------------------------------------------------------------
# Gold-annotation vocabulary
# ---------------------------------------------------------------------------

GOLD_REASONS: dict[str, str] = {
    EC.IMPERSONATION.value: "Caller claims institutional authority and uses it to justify the requests that follow.",
    EC.FALSE_PROBLEM.value: "Caller asserts an account problem the victim did not initiate, creating the emergency the instructions then 'solve'.",
    EC.FALSE_SOLUTION.value: "Caller presents themselves as the only remedy for the problem they described.",
    EC.THREAT.value: "Caller threatens loss of funds, blocking, or legal consequences to force compliance.",
    EC.FEAR.value: "Caller frames the victim's money as under active threat.",
    EC.URGENCY.value: "Caller imposes a deadline that removes time to verify independently.",
    EC.SECRECY.value: "Caller demands the conversation be kept secret, removing normal external checks.",
    EC.ISOLATION.value: "Caller prevents the victim from contacting the bank or hanging up to verify.",
    EC.OTP_REQUEST.value: "Caller asks the victim to disclose a one-time authentication code, which authorises a transaction.",
    EC.CARD_REQUEST.value: "Caller asks for full card credentials, which are sufficient to make payments.",
    EC.PASSWORD_REQUEST.value: "Caller asks for a password, login or PIN, granting standing account access.",
    EC.PERSONAL_DATA_REQUEST.value: "Caller collects identification data.",
    EC.MONEY_TRANSFER.value: "Caller instructs the victim to move or deposit funds.",
    EC.SAFE_ACCOUNT.value: "Caller directs funds to a 'safe / reserve / special' account, which does not exist at Kazakhstan banks.",
    EC.CRYPTO_TRANSFER.value: "Caller directs funds into crypto, making the transfer irreversible.",
    EC.REMOTE_ACCESS.value: "Caller seeks remote control of the victim's device.",
    EC.SCREEN_SHARING.value: "Caller asks the victim to share their screen, exposing balances and incoming codes.",
    EC.APP_INSTALLATION.value: "Caller asks the victim to install software supplied by the caller.",
    EC.PROTECTIVE_ADVICE.value: "Speaker warns against disclosure or points to an official channel — the opposite of a fraudulent request.",
}

RECOMMENDED_ACTION: dict[str, str] = {
    "SCAM": (
        "Terminate the call. Do not share codes, card data or passwords and do not move money. "
        "Contact the bank yourself using the number on the back of the card and report the call."
    ),
    "SUSPICIOUS": (
        "Do not act on the caller's instructions. End the call and verify independently through the "
        "official bank app or hotline."
    ),
    "SAFE": (
        "No fraud indicators requiring intervention. Standard advice applies: never share one-time "
        "codes, card details or passwords with anyone who calls you."
    ),
}

REQUESTED_ACTION_TEXT: dict[str, str] = {
    EC.OTP_REQUEST.value: "Dictate the one-time SMS/push confirmation code to the caller",
    EC.CARD_REQUEST.value: "Disclose full card number, expiry date or CVV/CVC",
    EC.PASSWORD_REQUEST.value: "Disclose account password, login or PIN",
    EC.PERSONAL_DATA_REQUEST.value: "Disclose personal identification data (IIN, ID document, date of birth)",
    EC.MONEY_TRANSFER.value: "Transfer or deposit funds as instructed by the caller",
    EC.SAFE_ACCOUNT.value: "Move funds to a so-called 'safe account' nominated by the caller",
    EC.CRYPTO_TRANSFER.value: "Send funds to a crypto wallet or trading account",
    EC.REMOTE_ACCESS.value: "Grant remote access to the victim's phone or computer",
    EC.SCREEN_SHARING.value: "Share the victim's screen during a banking session",
    EC.APP_INSTALLATION.value: "Install an application supplied by the caller",
    EC.ISOLATION.value: "Stay on the line and not contact the bank independently",
    EC.SECRECY.value: "Keep the conversation secret from family and bank staff",
}


# ---------------------------------------------------------------------------
# ASR noise (spec section 16)
# ---------------------------------------------------------------------------

NOISE_FILLERS = ("ээ", "эм", "ммм", "ну", "әм")
_VOWELS = "аеиоуыэюяәөүіұ"


def degrade(text: str, rng: random.Random, strength: float = 0.5) -> str:
    """Make text look like imperfect ASR output.

    Deliberately never removes a digit or a whole word — the model has to cope
    with noise, but the corpus must not lose the evidence it is annotated for.
    """
    words = text.split()
    out: list[str] = []
    for word in words:
        if rng.random() < 0.06 * strength:
            out.append(rng.choice(NOISE_FILLERS))
        if rng.random() < 0.05 * strength and len(word) > 3 and not any(c.isdigit() for c in word):
            out.append(f"{word[:2]}... {word}")  # stutter / restart
            continue
        if rng.random() < 0.07 * strength and len(word) > 5 and not any(c.isdigit() for c in word):
            position = next(
                (i for i, ch in enumerate(word[2:-1], start=2) if ch in _VOWELS), None
            )
            if position:
                word = word[:position] + word[position + 1 :]  # dropped vowel
        out.append(word)
    noisy = " ".join(out)
    if rng.random() < 0.7 * strength:
        noisy = noisy.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    if rng.random() < 0.4 * strength:
        noisy = noisy.lower()
    return noisy


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

LANGUAGE_MODES = ("ru", "kk", "mixed")


@dataclass
class RenderedCall:
    call_id: str
    family: Family
    language_mode: str
    noisy: bool
    turns: list[dict] = field(default_factory=list)

    @property
    def slices(self) -> list[str]:
        """Evaluation slices this conversation belongs to (spec section 25)."""
        names = [self.language_mode, self.family.difficulty]
        names.append("scam" if self.family.is_scam else "legitimate")
        if self.noisy:
            names.append("asr_noisy")
        if self.family.holdout:
            names.append("unseen_pattern")
        return list(dict.fromkeys(names))  # "legitimate" is both a difficulty and a label


def render_family(
    family: Family,
    *,
    language_mode: str,
    noisy: bool,
    seed: int,
    variant: int = 0,
) -> RenderedCall:
    """Produce one conversation: turn texts plus per-turn annotations."""
    rng = random.Random(f"{family.key}:{language_mode}:{noisy}:{seed}:{variant}")
    call_id = f"{family.key}_{language_mode}{'_noisy' if noisy else ''}_v{variant}"
    rendered = RenderedCall(call_id=call_id, family=family, language_mode=language_mode, noisy=noisy)

    cursor = 1.2
    for position, turn in enumerate(family.turns):
        if language_mode == "ru":
            text = turn.ru
        elif language_mode == "kk":
            text = turn.kk
        else:
            # Intra-sentential switch where the script provides one; otherwise
            # switch language between turns, which is equally natural here.
            text = turn.mixed or (turn.ru if position % 2 else turn.kk)
        if noisy:
            text = degrade(text, rng, strength=0.6 if turn.events else 0.45)

        duration = max(2.4, min(11.0, len(text) / 13.5))
        start = round(cursor, 1)
        end = round(start + duration, 1)
        confidence = 1.0
        if noisy:
            confidence = round(rng.uniform(0.42, 0.74), 2)
        else:
            confidence = round(rng.uniform(0.88, 0.99), 2)

        rendered.turns.append(
            {
                "speaker": turn.speaker.value,
                "stage": turn.stage,
                "text": text,
                "start": start,
                "end": end,
                "confidence": confidence,
                "events": [
                    {"category": category, "severity": severity} for category, severity in turn.events
                ],
            }
        )
        cursor = end + round(rng.uniform(0.4, 2.2), 1)
    return rendered


def iter_corpus(
    families: Sequence[Family] | None = None,
    *,
    seed: int = 20260809,
    variants: int = 2,
    include_noisy: bool = True,
) -> Iterator[RenderedCall]:
    """Every (family × language × variant × noise) conversation."""
    pool = families if families is not None else ALL_FAMILIES
    for family in pool:
        for language_mode in LANGUAGE_MODES:
            for variant in range(variants):
                yield render_family(
                    family, language_mode=language_mode, noisy=False, seed=seed, variant=variant
                )
                if include_noisy:
                    yield render_family(
                        family, language_mode=language_mode, noisy=True, seed=seed, variant=variant
                    )


# ---------------------------------------------------------------------------
# Gold analysis derived from the script + the rendered text
# ---------------------------------------------------------------------------


def gold_analysis(rendered: RenderedCall, *, texts: Sequence[str], timestamps: Sequence[str]) -> dict:
    """The target JSON for one conversation (spec section 17).

    `texts`/`timestamps` come from the **built transcript**, so every evidence
    string is verbatim what the model will see in its prompt.
    """
    family = rendered.family
    risk_factors: list[dict] = []
    categories: list[str] = []
    for index, turn in enumerate(rendered.turns):
        for event in turn["events"]:
            categories.append(event["category"])
            risk_factors.append(
                {
                    "timestamp": timestamps[index],
                    "speaker": turn["speaker"],
                    "category": event["category"],
                    "severity": event["severity"],
                    "evidence": texts[index],
                    "reason": GOLD_REASONS.get(event["category"], ""),
                }
            )

    # Impersonation is annotated on the *identity claim* turn, but only for scams:
    # a legitimate introduction is not impersonation (spec section 9).
    if family.is_scam:
        for index, turn in enumerate(rendered.turns):
            if turn["stage"] == SG.IDENTITY_CLAIM.value and turn["speaker"] == Speaker.CALLER.value:
                risk_factors.insert(
                    0,
                    {
                        "timestamp": timestamps[index],
                        "speaker": turn["speaker"],
                        "category": EC.IMPERSONATION.value,
                        "severity": SV.HIGH.value,
                        "evidence": texts[index],
                        "reason": GOLD_REASONS[EC.IMPERSONATION.value],
                    },
                )
                categories.append(EC.IMPERSONATION.value)
                break

    tactics: list[str] = []
    for category in categories:
        tactic = EVENT_TO_TACTIC.get(category)
        if tactic and tactic not in tactics:
            tactics.append(tactic)
    if set(family.scam_types) & IMPERSONATION_SCAM_TYPES and "AUTHORITY" not in tactics:
        tactics.append("AUTHORITY")

    requested = []
    for category in categories:
        text = REQUESTED_ACTION_TEXT.get(category)
        if text and text not in requested:
            requested.append(text)

    stage = max(
        (turn["stage"] for turn in rendered.turns),
        key=lambda value: STAGE_DEPTH.get(value, 0),
        default=SG.UNKNOWN.value,
    )

    confidence = 0.9 if family.difficulty == "obvious" else 0.74
    if not family.is_scam:
        confidence = 0.88
    if rendered.noisy:
        confidence = round(confidence - 0.08, 2)

    return {
        "classification": family.classification,
        "confidence": confidence,
        "scam_types": list(family.scam_types),
        "tactics": tactics,
        "conversation_stage": stage,
        "requested_actions": requested,
        "risk_factors": risk_factors,
        "explanation": _gold_explanation(rendered, risk_factors),
        "recommended_action": RECOMMENDED_ACTION[family.classification],
    }


def _gold_explanation(rendered: RenderedCall, risk_factors: Sequence[dict]) -> str:
    family = rendered.family
    if not family.is_scam:
        protective = [f for f in risk_factors if f["category"] == EC.PROTECTIVE_ADVICE.value]
        parts = [
            "No social-engineering sequence is present.",
            f"This is a {family.label.lower()}.",
        ]
        if family.call_direction == "inbound":
            parts.append("The customer placed the call and set the topic.")
        if protective:
            parts.append(
                f"At {protective[0]['timestamp']} a speaker explicitly warns against disclosing "
                "credentials or points to an official channel."
            )
        parts.append(
            "Banking vocabulary alone — bank names, cards, transfers, codes, loans — is not evidence "
            "of fraud: the caller makes no request for one-time codes, card credentials, passwords, "
            "money movement or device access."
        )
        return " ".join(parts)

    steps: list[str] = []
    for category in (
        EC.IMPERSONATION.value, EC.FALSE_PROBLEM.value, EC.THREAT.value, EC.FEAR.value,
        EC.URGENCY.value, EC.SECRECY.value, EC.ISOLATION.value, EC.FALSE_SOLUTION.value,
    ):
        match = next((f for f in risk_factors if f["category"] == category), None)
        if match:
            steps.append(f'{category.lower().replace("_", " ")} at {match["timestamp"]}')
    payoff = [f for f in risk_factors if f["category"] in HARMFUL_ACTION_EVENTS]
    for factor in payoff[:3]:
        steps.append(
            f'{factor["category"].lower().replace("_", " ")} at {factor["timestamp"]} '
            f'("{factor["evidence"][:80]}")'
        )
    return (
        f"The caller follows a recognisable social-engineering sequence — {'; then '.join(steps)}. "
        "The requested action hands control of the victim's money, credentials or device to the "
        "caller, which is the payoff step of the attack."
    )
