"""
Benchmark datasets for Indic language translation evaluation.

Contains curated test sets with reference translations across multiple
domains and difficulty categories for the seven supported Indic languages.

Curated content
---------------
* 42 English source sentences with references in all 7 Indic languages:
  - domains: general, technical, medical, financial, academic,
             education, government, legal, conversational
  - difficulty categories: simple, medium, complex, idiom, names,
             numbers, dates, technical, long, colloquial
* 25 code-mixed sentences (Indic script + English words) across all 7
  supported Indic languages, with English references, used for the
  Code-Mixed Translation Accuracy benchmark.

NOTE ON REFERENCES
------------------
The curated references were machine-generated and lightly edited. Before
using them for a publication, verify them with a native speaker (or swap
in FLORES-200 / IN22 references via `load_flores_dataset`).

External datasets
-----------------
`load_flores_dataset()` optionally loads FLORES-200 (facebook/flores,
config "flores200") from HuggingFace — the research-grade parallel corpus
used by the WMT/FLORES benchmarks. Requires: pip install datasets
"""

import os
from typing import List, Dict, Optional

###############################################################################
# Language metadata
###############################################################################

# FLORES-200 codes for the languages we support (used by load_flores_dataset)
FLORES_CODES = {
    'english': 'eng_Latn',
    'hindi': 'hin_Deva',
    'marathi': 'mar_Deva',
    'bengali': 'ben_Beng',
    'gujarati': 'guj_Gujr',
    'tamil': 'tam_Taml',
    'telugu': 'tel_Telu',
    'kannada': 'kan_Knda',
}

LANGUAGE_META = {
    'hindi': {'name': 'Hindi', 'native': 'हिंदी', 'code': 'hin_Deva', 'flores_code': 'hin_Deva'},
    'bengali': {'name': 'Bengali', 'native': 'বাংলা', 'code': 'ben_Beng', 'flores_code': 'ben_Beng'},
    'marathi': {'name': 'Marathi', 'native': 'मराठी', 'code': 'mar_Deva', 'flores_code': 'mar_Deva'},
    'gujarati': {'name': 'Gujarati', 'native': 'ગુજરાતી', 'code': 'guj_Gujr', 'flores_code': 'guj_Gujr'},
    'tamil': {'name': 'Tamil', 'native': 'தமிழ்', 'code': 'tam_Taml', 'flores_code': 'tam_Taml'},
    'telugu': {'name': 'Telugu', 'native': 'తెలుగు', 'code': 'tel_Telu', 'flores_code': 'tel_Telu'},
    'kannada': {'name': 'Kannada', 'native': 'ಕನ್ನಡ', 'code': 'kan_Knda', 'flores_code': 'kan_Knda'},
}

# Curated reference languages (all languages we can provide references for)
CURATED_LANGS = ['hindi', 'marathi', 'bengali', 'gujarati', 'tamil', 'telugu', 'kannada']


###############################################################################
# Curated quality benchmark (domains + difficulty)
#
# Each entry: (domain, category, english_sentence, {lang: reference})
###############################################################################

CURATED_ITEMS = [
    # --------------------------- General -----------------------------------
    ('general', 'simple', "The weather is beautiful today and I want to go for a walk in the park.",
     {'hindi': "आज मौसम बहुत सुंदर है और मैं पार्क में टहलने जाना चाहता हूँ।",
      'bengali': "আজ আবহাওয়া খুব সুন্দর এবং আমি পার্কে হাঁটতে যেতে চাই।",
      'marathi': "आज हवामान खूप सुंदर आहे आणि मला उद्यानात फिरायला जायचे आहे.",
      'gujarati': "આજે હવામાન ખૂબ સુંદર છે અને હું પાર્કમાં ફરવા જવા માંગુ છું.",
      'tamil': "இன்று வானிலை மிகவும் அழகாக உள்ளது மற்றும் நான் பூங்காவில் நடந்து செல்ல விரும்புகிறேன்.",
      'telugu': "ఈ రోజు వాతావరణం చాలా అందంగా ఉంది మరియు నేను పార్కులో నడవడానికి వెళ్లాలనుకుంటున్నాను.",
      'kannada': "ಇಂದು ಹವಾಮಾನ ಬಹಳ ಸುಂದರವಾಗಿದೆ ಮತ್ತು ನಾನು ಉದ್ಯಾನವನದಲ್ಲಿ ವಾಕ್ ಮಾಡಲು ಹೋಗಲು ಬಯಸುತ್ತೇನೆ."}),
    ('general', 'simple', "Please tell me where I can find the nearest hospital.",
     {'hindi': "कृपया मुझे बताएं कि मैं निकटतम अस्पताल कहां पा सकता हूं।",
      'bengali': "অনুগ্রহ করে আমাকে বলুন নিকটস্থ হাসপাতাল কোথায় পাব।",
      'marathi': "कृपया मला सांगा की मी जवळचे रुग्णालय कुठे शोधू शकतो.",
      'gujarati': "કૃપા કરીને મને કહો કે હું નજીકનું હોસ્પિટલ ક્યાં શોધી શકું.",
      'tamil': "அருகிலுள்ள மருத்துவமனையை நான் எங்கே காணலாம் என்று தயவுசெய்து சொல்லுங்கள்.",
      'telugu': "దయచేసి నాకు దగ్గరలోని ఆసుపత్రి ఎక్కడ దొరుకుతుందో చెప్పండి.",
      'kannada': "ದಯವಿಟ್ಟು ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ ಎಂದು ನನಗೆ ತಿಳಿಸಿ."}),
    ('general', 'simple', "I would like to order two cups of tea and some snacks.",
     {'hindi': "मैं दो कप चाय और कुछ नाश्ता ऑर्डर करना चाहूंगा।",
      'bengali': "আমি দুই কাপ চা এবং কিছু নাস্তা অর্ডার করতে চাই।",
      'marathi': "मला दोन कप चहा आणि काही स्नॅक्स ऑर्डर करायचे आहेत.",
      'gujarati': "હું બે કપ ચા અને થોડો નાસ્તો ઓર્ડર કરવા માંગુ છું.",
      'tamil': "நான் இரண்டு கப் தேநீர் மற்றும் சில சிற்றுண்டிகள் ஆர்டர் செய்ய விரும்புகிறேன்.",
      'telugu': "నేను రెండు కప్పుల టీ మరియు కొన్ని స్నాక్స్ ఆర్డర్ చేయాలనుకుంటున్నాను.",
      'kannada': "ನಾನು ಎರಡು ಕಪ್ ಚಹಾ ಮತ್ತು ಕೆಲವು ತಿಂಡಿಗಳನ್ನು ಆರ್ಡರ್ ಮಾಡಲು ಬಯಸುತ್ತೇನೆ."}),
    ('general', 'medium', "Can you help me understand this document?",
     {'hindi': "क्या आप मुझे इस दस्तावेज़ को समझने में मदद कर सकते हैं?",
      'bengali': "আপনি কি আমাকে এই ডকুমেন্ট বুঝতে সাহায্য করতে পারেন?",
      'marathi': "कृपया तुम्ही मला हा दस्तऐवज समजून घेण्यास मदत करू शकता का?",
      'gujarati': "શું તમે મને આ દસ્તાવેજ સમજવામાં મદદ કરી શકશો?",
      'tamil': "இந்த ஆவணத்தை புரிந்துகொள்ள நீங்கள் எனக்கு உதவ முடியுமா?",
      'telugu': "మీరు నాకు ఈ డాక్యుమెంట్ అర్థం చేసుకోవడానికి సహాయం చేయగలరా?",
      'kannada': "ಈ ಡಾಕ್ಯುಮೆಂಟ್ ಅನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ನೀವು ನನಗೆ ಸಹಾಯ ಮಾಡಬಹುದೇ?"}),

    # --------------------------- Technical ---------------------------------
    ('technical', 'technical', "The server processes data using a neural network architecture.",
     {'hindi': "सर्वर न्यूरल नेटवर्क आर्किटेक्चर का उपयोग करके डेटा प्रोसेस करता है।",
      'bengali': "সার্ভার নিউরাল নেটওয়ার্ক আর্কিটেকচার ব্যবহার করে ডেটা প্রসেস করে।",
      'marathi': "सर्व्हर न्यूरल नेटवर्क आर्किटेक्चर वापरून डेटा प्रक्रिया करतो.",
      'gujarati': "સર્વર ન્યુરલ નેટવર્ક આર્કિટેક્ચરનો ઉપયોગ કરીને ડેટા પ્રોસેસ કરે છે.",
      'tamil': "சர்வர் நரம்பியல் வலையமைப்பு கட்டமைப்பைப் பயன்படுத்தி தரவை செயலாக்குகிறது.",
      'telugu': "సర్వర్ న్యూరల్ నెట్వర్క్ ఆర్కిటెక్చర్ ఉపయోగించి డేటాను ప్రాసెస్ చేస్తుంది.",
      'kannada': "ಸರ್ವರ್ ನ್ಯೂರಲ್ ನೆಟ್ವರ್ಕ್ ಆರ್ಕಿಟೆಕ್ಚರ್ ಬಳಸಿ ಡೇಟಾವನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸುತ್ತದೆ."}),
    ('technical', 'technical', "Please install the latest software update to fix the security vulnerability.",
     {'hindi': "कृपया सुरक्षा कमजोरी को ठीक करने के लिए नवीनतम सॉफ्टवेयर अपडेट इंस्टॉल करें।",
      'bengali': "নিরাপত্তা দুর্বলতা ঠিক করতে সর্বশেষ সফটওয়্যার আপডেট ইনস্টল করুন।",
      'marathi': "सुरक्षा भेद्यता दुरुस्त करण्यासाठी कृपया नवीनतम सॉफ्टवेअर अपडेट स्थापित करा.",
      'gujarati': "સુરક્ષા નબળાઈ સુધારવા માટે કૃપા કરીને નવીનતમ સૉફ્ટવેર અપડેટ ઇન્સ્ટોલ કરો.",
      'tamil': "பாதுகாப்பு பலவீனத்தை சரிசெய்ய சமீபத்திய மென்பொருள் புதுப்பிப்பை நிறுவவும்.",
      'telugu': "దయచేసి భద్రతా లోపాన్ని పరిష్కరించడానికి తాజా సాఫ్ట్వేర్ అప్డేట్ను ఇన్స్టాల్ చేయండి.",
      'kannada': "ಭದ್ರತಾ ದುರ್ಬಲತೆಯನ್ನು ಸರಿಪಡಿಸಲು ದಯವಿಟ್ಟು ಇತ್ತೀಚಿನ ಸಾಫ್ಟ್ವೇರ್ ಅಪ್ಡೇಟ್ ಅನ್ನು ಸ್ಥಾಪಿಸಿ."}),
    ('technical', 'technical', "Machine learning models require large amounts of training data.",
     {'hindi': "मशीन लर्निंग मॉडल को बड़ी मात्रा में प्रशिक्षण डेटा की आवश्यकता होती है।",
      'bengali': "মেশিন লার্নিং মডেলের জন্য প্রচুর পরিমাণে প্রশিক্ষণ ডেটা প্রয়োজন।",
      'marathi': "मशीन लर्निंग मॉडेल्सना मोठ्या प्रमाणात प्रशिक्षण डेटाची आवश्यकता असते.",
      'gujarati': "મશીન લર્નિંગ મૉડલને મોટા પ્રમાણમાં તાલીમ ડેટાની જરૂર પડે છે.",
      'tamil': "இயந்திர கற்றல் மாதிரிகளுக்கு அதிக அளவு பயிற்சி தரவு தேவைப்படுகிறது.",
      'telugu': "మెషిన్ లెర్నింగ్ మోడల్లకు పెద్ద మొత్తంలో శిక్షణ డేటా అవసరం.",
      'kannada': "ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಮಾಡೆಲ್‌ಗಳಿಗೆ ದೊಡ್ಡ ಪ್ರಮಾಣದ ತರಬೇತಿ ಡೇಟಾ ಅಗತ್ಯವಿದೆ."}),
    ('technical', 'medium', "The database query returned results in under fifty milliseconds.",
     {'hindi': "डेटाबेस क्वेरी ने पचास मिलीसेकंड से कम समय में परिणाम लौटाए।",
      'bengali': "ডেটাবেস কোয়েরি পঞ্চাশ মিলিসেকেন্ডেরও কম সময়ে ফলাফল ফিরিয়ে দিয়েছে।",
      'marathi': "डेटाबेस क्वेरीने पन्नास मिलिसेकंदांपेक्षा कमी वेळात परिणाम परत केले.",
      'gujarati': "ડેટાબેઝ ક્વેરીએ પચાસ મિલિસેકન્ડથી ઓછા સમયમાં પરિણામો પરત કર્યા.",
      'tamil': "தரவுத்தள வினவல் ஐம்பது மில்லிசெகண்டுகளுக்குள் முடிவுகளை வழங்கியது.",
      'telugu': "డేటాబేస్ క్వేరీ యాభై మిల్లీసెకన్లలోపు ఫలితాలను అందించింది.",
      'kannada': "ಡೇಟಾಬೇಸ್ ಕ್ವೆರಿ ಐವತ್ತು ಮಿಲಿಸೆಕೆಂಡ್‌ಗಳಿಗಿಂತ ಕಡಿಮೆ ಸಮಯದಲ್ಲಿ ಫಲಿತಾಂಶಗಳನ್ನು ನೀಡಿತು."}),

    # --------------------------- Medical -----------------------------------
    ('medical', 'medium', "The patient should take this medicine twice a day after meals.",
     {'hindi': "रोगी को यह दवा भोजन के बाद दिन में दो बार लेनी चाहिए।",
      'bengali': "রোগীর এই ওষুধ খাবারের পর দিনে দুবার খাওয়া উচিত।",
      'marathi': "रुग्णाने हे औषध जेवणानंतर दिवसातून दोनदा घ्यावे.",
      'gujarati': "દર્દીએ આ દવા ભોજન પછી દિવસમાં બે વાર લેવી જોઈએ.",
      'tamil': "நோயாளி இந்த மருந்தை உணவுக்குப் பிறகு ஒரு நாளைக்கு இரண்டு முறை எடுக்க வேண்டும்.",
      'telugu': "రోగి ఈ మందును భోజనం తర్వాత రోజుకు రెండుసార్లు తీసుకోవాలి.",
      'kannada': "ರೋಗಿಯು ಈ ಔಷಧಿಯನ್ನು ಊಟದ ನಂತರ ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ ಸೇವಿಸಬೇಕು."}),
    ('medical', 'medium', "Regular exercise and a balanced diet are important for good health.",
     {'hindi': "अच्छे स्वास्थ्य के लिए नियमित व्यायाम और संतुलित आहार महत्वपूर्ण हैं।",
      'bengali': "সুস্থ থাকার জন্য নিয়মিত ব্যায়াম এবং সুষম খাদ্য গুরুত্বপূর্ণ।",
      'marathi': "चांगल्या आरोग्यासाठी नियमित व्यायाम आणि संतुलित आहार महत्त्वाचे आहेत.",
      'gujarati': "સારા સ્વાસ્થ્ય માટે નિયમિત કસરત અને સંતુલિત આહાર મહત્વપૂર્ણ છે.",
      'tamil': "நல்ல ஆரோக்கியத்திற்கு வழக்கமான உடற்பயிற்சி மற்றும் சீரான உணவு முக்கியம்.",
      'telugu': "మంచి ఆరోగ్యానికి క్రమం తప్పకుండా వ్యాయామం మరియు సమతుల్య ఆహారం ముఖ్యమైనవి.",
      'kannada': "ಉತ್ತಮ ಆರೋಗ್ಯಕ್ಕಾಗಿ ನಿಯಮಿತ ವ್ಯಾಯಾಮ ಮತ್ತು ಸಮತೋಲಿತ ಆಹಾರ ಮುಖ್ಯವಾಗಿದೆ."}),
    ('medical', 'complex', "The doctor recommended a blood test to check for vitamin deficiency.",
     {'hindi': "डॉक्टर ने विटामिन की कमी की जांच के लिए रक्त परीक्षण की सिफारिश की।",
      'bengali': "ডাক্তার ভিটামিনের অভাব পরীক্ষা করার জন্য রক্ত পরীক্ষার পরামর্শ দিয়েছেন।",
      'marathi': "डॉक्टरांनी जीवनसत्वाच्या कमतरतेची तपासणी करण्यासाठी रक्त तपासणीची शिफारस केली.",
      'gujarati': "ડૉક્ટરે વિટામિનની ઉણપ ચકાસવા માટે રક્ત પરીક્ષણની ભલામણ કરી.",
      'tamil': "வைட்டமின் குறைபாட்டை சரிபார்க்க மருத்துவர் இரத்த பரிசோதனையை பரிந்துரைத்தார்.",
      'telugu': "విటమిన్ లోపాన్ని తనిఖీ చేయడానికి డాక్టర్ రక్త పరీక్షను సిఫార్సు చేశారు.",
      'kannada': "ವಿಟಮಿನ್ ಕೊರತೆಯನ್ನು ಪರೀಕ್ಷಿಸಲು ವೈದ್ಯರು ರಕ್ತ ಪರೀಕ್ಷೆಯನ್ನು ಶಿಫಾರಸು ಮಾಡಿದರು."}),
    ('medical', 'medium', "Please wash your hands thoroughly to prevent the spread of infection.",
     {'hindi': "संक्रमण के प्रसार को रोकने के लिए कृपया अपने हाथों को अच्छी तरह धोएं।",
      'bengali': "সংক্রমণ ছড়ানো প্রতিরোধ করতে অনুগ্রহ করে ভালোভাবে হাত ধুয়ে নিন।",
      'marathi': "संसर्ग पसरू नये म्हणून कृपया आपले हात चांगले धुवा.",
      'gujarati': "ચેપ ફેલાવો અટકાવવા માટે કૃપા કરીને તમારા હાથ સારી રીતે ધોઈ લો.",
      'tamil': "நோய்த்தொற்று பரவாமல் தடுக்க உங்கள் கைகளை நன்கு கழுவவும்.",
      'telugu': "సంక్రమణ వ్యాప్తిని నివారించడానికి దయచేసి మీ చేతులను బాగా కడగాలి.",
      'kannada': "ಸೋಂಕು ಹರಡುವುದನ್ನು ತಡೆಯಲು ದಯವಿಟ್ಟು ನಿಮ್ಮ ಕೈಗಳನ್ನು ಚೆನ್ನಾಗಿ ತೊಳೆಯಿರಿ."}),

    # --------------------------- Financial --------------------------------
    ('financial', 'medium', "The bank offers a low interest rate on education loans.",
     {'hindi': "बैंक शिक्षा ऋण पर कम ब्याज दर प्रदान करता है।",
      'bengali': "ব্যাংক শিক্ষা ঋণের উপর কম সুদের হার অফার করে।",
      'marathi': "बँक शिक्षण कर्जावर कमी व्याजदर देते.",
      'gujarati': "બેંક શિક્ષણ લોન પર ઓછો વ્યાજ દર આપે છે.",
      'tamil': "வங்கி கல்விக் கடனுக்கு குறைந்த வட்டி விகிதத்தை வழங்குகிறது.",
      'telugu': "బ్యాంకు విద్యా రుణాలపై తక్కువ వడ్డీ రేటును అందిస్తుంది.",
      'kannada': "ಬ್ಯಾಂಕ್ ಶಿಕ್ಷಣ ಸಾಲದ ಮೇಲೆ ಕಡಿಮೆ ಬಡ್ಡಿ ದರವನ್ನು ನೀಡುತ್ತದೆ."}),
    ('financial', 'medium', "Please transfer five thousand rupees to my savings account.",
     {'hindi': "कृपया मेरे बचत खाते में पांच हजार रुपये ट्रांसफर करें।",
      'bengali': "অনুগ্রহ করে আমার সঞ্চয় অ্যাকাউন্টে পাঁচ হাজার টাকা স্থানান্তর করুন।",
      'marathi': "कृपया माझ्या बचत खात्यात पाच हजार रुपये हस्तांतरित करा.",
      'gujarati': "કૃપા કરીને મારા બચત ખાતામાં પાંચ હજાર રૂપિયા ટ્રાન્સફર કરો.",
      'tamil': "தயவுசெய்து எனது சேமிப்புக் கணக்கில் ஐந்தாயிரம் ரூபாயை மாற்றவும்.",
      'telugu': "దయచేసి నా పొదుపు ఖాతాలో ఐదు వేల రూపాయలను బదిలీ చేయండి.",
      'kannada': "ದಯವಿಟ್ಟು ನನ್ನ ಉಳಿತಾಯ ಖಾತೆಗೆ ಐದು ಸಾವಿರ ರೂಪಾಯಿಗಳನ್ನು ವರ್ಗಾಯಿಸಿ."}),
    ('financial', 'complex', "The annual budget was approved by the finance committee yesterday.",
     {'hindi': "वार्षिक बजट को कल वित्त समिति द्वारा अनुमोदित किया गया।",
      'bengali': "বার্ষিক বাজেট গতকাল অর্থ কমিটি দ্বারা অনুমোদিত হয়েছে।",
      'marathi': "वार्षिक अर्थसंकल्प काल वित्त समितीने मंजूर केला.",
      'gujarati': "વાર્ષિક બજેટ ગઈકાલે નાણા સમિતિ દ્વારા મંજૂર કરવામાં આવ્યું.",
      'tamil': "வருடாந்திர பட்ஜெட் நேற்று நிதிக் குழுவால் அங்கீகரிக்கப்பட்டது.",
      'telugu': "వార్షిక బడ్జెట్ నిన్న ఫైనాన్స్ కమిటీచే ఆమోదించబడింది.",
      'kannada': "ವಾರ್ಷಿಕ ಬಜೆಟ್ ಅನ್ನು ನಿನ್ನೆ ಹಣಕಾಸು ಸಮಿತಿಯು ಅನುಮೋದಿಸಿತು."}),
    ('financial', 'complex', "Investors are concerned about the fluctuation in currency exchange rates.",
     {'hindi': "निवेशक मुद्रा विनिमय दरों में उतार-चढ़ाव को लेकर चिंतित हैं।",
      'bengali': "মুদ্রা বিনিময় হারের ওঠানামা নিয়ে বিনিয়োগকারীরা উদ্বিগ্ন।",
      'marathi': "चलन विनिमय दरातील चढउताराबद्दल गुंतवणूकदार चिंतित आहेत.",
      'gujarati': "રોકાણકારો ચલણ વિનિમય દરોમાં વધઘટ વિશે ચિંતિત છે.",
      'tamil': "நாணய மாற்று விகிதங்களில் ஏற்படும் ஏற்ற இறக்கம் குறித்து முதலீட்டாளர்கள் கவலை கொண்டுள்ளனர்.",
      'telugu': "కరెన్సీ మారకపు రేట్లలో హెచ్చుతగ్గుల గురించి పెట్టుబడిదారులు ఆందోళన చెందుతున్నారు.",
      'kannada': "ಕರೆನ್ಸಿ ವಿನಿಮಯ ದರಗಳಲ್ಲಿ ಏರಿಳಿತದ ಬಗ್ಗೆ ಹೂಡಿಕೆದಾರರು ಕಳವಳ ವ್ಯಕ್ತಪಡಿಸಿದ್ದಾರೆ."}),

    # --------------------------- Academic ----------------------------------
    ('academic', 'complex', "The research paper presents a novel approach to language translation.",
     {'hindi': "यह शोध पत्र भाषा अनुवाद के लिए एक नवीन दृष्टिकोण प्रस्तुत करता है।",
      'bengali': "গবেষণাপত্রটি ভাষা অনুবাদের জন্য একটি অভিনব পদ্ধতি উপস্থাপন করে।",
      'marathi': "संशोधन पत्र भाषा अनुवादासाठी एक नवीन दृष्टिकोन सादर करते.",
      'gujarati': "સંશોધન પેપર ભાષા અનુવાદ માટે એક નવીન અભિગમ રજૂ કરે છે.",
      'tamil': "ஆராய்ச்சி கட்டுரை மொழி மொழிபெயர்ப்பிற்கான ஒரு புதுமையான அணுகுமுறையை முன்வைக்கிறது.",
      'telugu': "పరిశోధన పత్రం భాషా అనువాదానికి ఒక నవల విధానాన్ని ప్రదర్శిస్తుంది.",
      'kannada': "ಸಂಶೋಧನಾ ಪ್ರಬಂಧವು ಭಾಷಾ ಅನುವಾದಕ್ಕೆ ಒಂದು ನವೀನ ವಿಧಾನವನ್ನು ಪ್ರಸ್ತುತಪಡಿಸುತ್ತದೆ."}),
    ('academic', 'medium', "Students must submit their assignments before the end of this week.",
     {'hindi': "छात्रों को इस सप्ताह के अंत से पहले अपना असाइनमेंट जमा करना होगा।",
      'bengali': "ছাত্রদের এই সপ্তাহ শেষ হওয়ার আগে তাদের অ্যাসাইনমেন্ট জমা দিতে হবে।",
      'marathi': "विद्यार्थ्यांनी या आठवड्याच्या अखेरीपूर्वी आपली असाइनमेंट सबमिट करणे आवश्यक आहे.",
      'gujarati': "વિદ્યાર્થીઓએ આ અઠવાડિયાના અંત પહેલા તેમની સોંપણીઓ સબમિટ કરવી આવશ્યક છે.",
      'tamil': "மாணவர்கள் இந்த வார இறுதிக்குள் தங்கள் பணிகளைச் சமர்ப்பிக்க வேண்டும்.",
      'telugu': "విద్యార్థులు ఈ వారం చివరిలోపు తమ అసైన్మెంట్లను సమర్పించాలి.",
      'kannada': "ವಿದ್ಯಾರ್ಥಿಗಳು ಈ ವಾರದ ಅಂತ್ಯದೊಳಗೆ ತಮ್ಮ ನಿಯೋಜನೆಗಳನ್ನು ಸಲ್ಲಿಸಬೇಕು."}),
    ('academic', 'medium', "The university library has thousands of books on artificial intelligence.",
     {'hindi': "विश्वविद्यालय पुस्तकालय में कृत्रिम बुद्धिमत्ता पर हजारों पुस्तकें हैं।",
      'bengali': "বিশ্ববিদ্যালয়ের গ্রন্থাগারে কৃত্রিম বুদ্ধিমত্তার উপর হাজার হাজার বই আছে।",
      'marathi': "विद्यापीठाच्या ग्रंथालयात कृत्रिम बुद्धिमत्तेवर हजारो पुस्तके आहेत.",
      'gujarati': "યુનિવર્સિટી પુસ્તકાલયમાં કૃત્રિમ બુદ્ધિમત્તા પર હજારો પુસ્તકો છે.",
      'tamil': "பல்கலைக்கழக நூலகத்தில் செயற்கை நுண்ணறிவு பற்றி ஆயிரக்கணக்கான புத்தகங்கள் உள்ளன.",
      'telugu': "విశ్వవిద్యాలయ గ్రంథాలయంలో కృత్రిమ మేధస్సుపై వేలాది పుస్తకాలు ఉన్నాయి.",
      'kannada': "ವಿಶ್ವವಿದ್ಯಾಲಯದ ಗ್ರಂಥಾಲಯದಲ್ಲಿ ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆಯ ಮೇಲೆ ಸಾವಿರಾರು ಪುಸ್ತಕಗಳಿವೆ."}),
    ('academic', 'complex', "The experiment results confirm the hypothesis proposed by the researcher.",
     {'hindi': "प्रयोग के परिणाम शोधकर्ता द्वारा प्रस्तावित परिकल्पना की पुष्टि करते हैं।",
      'bengali': "পরীক্ষার ফলাফল গবেষক দ্বারা প্রস্তাবিত অনুমান নিশ্চিত করে।",
      'marathi': "प्रयोगाचे परिणाम संशोधकाने प्रस्तावित केलेल्या गृहीतकाची पुष्टी करतात.",
      'gujarati': "પ્રયોગના પરિણામો સંશોધક દ્વારા પ્રસ્તાવિત પૂર્વધારણાની પુષ્ટિ કરે છે.",
      'tamil': "பரிசோதனை முடிவுகள் ஆராய்ச்சியாளரால் முன்மொழியப்பட்ட கருதுகோளை உறுதிப்படுத்துகின்றன.",
      'telugu': "ప్రయోగ ఫలితాలు పరిశోధకుడు ప్రతిపాదించిన పరికల్పనను నిర్ధారిస్తాయి.",
      'kannada': "ಪ್ರಯೋಗದ ಫಲಿತಾಂಶಗಳು ಸಂಶೋಧಕರು ಪ್ರಸ್ತಾಪಿಸಿದ ಊಹೆಯನ್ನು ದೃಢೀಕರಿಸುತ್ತವೆ."}),

    # --------------------------- Education --------------------------------
    ('education', 'medium', "Students should submit their assignments before Friday.",
     {'hindi': "छात्रों को अपना असाइनमेंट शुक्रवार से पहले जमा करना चाहिए।",
      'bengali': "শিক্ষার্থীদের শুক্রবারের আগে তাদের অ্যাসাইনমেন্ট জমা দেওয়া উচিত।",
      'marathi': "विद्यार्थ्यांनी आपली असाइनमेंट शुक्रवारपूर्वी सबमिट करावी.",
      'gujarati': "વિદ્યાર્થીઓએ શુક્રવાર પહેલા તેમની સોંપણીઓ સબમિટ કરવી જોઈએ.",
      'tamil': "மாணவர்கள் வெள்ளிக்கிழமைக்கு முன் தங்கள் பணிகளைச் சமர்ப்பிக்க வேண்டும்.",
      'telugu': "విద్యార్థులు శుక్రవారానికి ముందు తమ అసైన్మెంట్లను సమర్పించాలి.",
      'kannada': "ವಿದ್ಯಾರ್ಥಿಗಳು ಶುಕ್ರವಾರದ ಮೊದಲು ತಮ್ಮ ನಿಯೋಜನೆಗಳನ್ನು ಸಲ್ಲಿಸಬೇಕು."}),
    ('education', 'simple', "The teacher explained the new lesson to the class.",
     {'hindi': "शिक्षक ने कक्षा को नया पाठ समझाया।",
      'bengali': "শিক্ষক শ্রেণিকক্ষে নতুন পাঠটি ব্যাখ্যা করলেন।",
      'marathi': "शिक्षकाने वर्गाला नवीन धडा समजावून सांगितला.",
      'gujarati': "શિક્ષકે વર્ગને નવો પાઠ સમજાવ્યો.",
      'tamil': "ஆசிரியர் வகுப்பிற்கு புதிய பாடத்தை விளக்கினார்.",
      'telugu': "ఉపాధ్యాయుడు తరగతికి కొత్త పాఠాన్ని వివరించాడు.",
      'kannada': "ಶಿಕ್ಷಕರು ತರಗತಿಗೆ ಹೊಸ ಪಾಠವನ್ನು ವಿವರಿಸಿದರು."}),
    ('education', 'medium', "She scored the highest marks in the final examination.",
     {'hindi': "उसने अंतिम परीक्षा में सर्वोच्च अंक प्राप्त किए।",
      'bengali': "তিনি চূড়ান্ত পরীক্ষায় সর্বোচ্চ নম্বর পেয়েছেন।",
      'marathi': "तिने अंतिम परीक्षेत सर्वोच्च गुण मिळवले.",
      'gujarati': "તેણે અંતિમ પરીક્ષામાં સર્વોચ્ચ ગુણ મેળવ્યા.",
      'tamil': "அவர் இறுதித் தேர்வில் அதிக மதிப்பெண்கள் பெற்றார்.",
      'telugu': "ఆమె చివరి పరీక్షలో అత్యధిక మార్కులు సాధించింది.",
      'kannada': "ಅವರು ಅಂತಿಮ ಪರೀಕ್ಷೆಯಲ್ಲಿ ಅತ್ಯಧಿಕ ಅಂಕಗಳನ್ನು ಪಡೆದರು."}),

    # --------------------------- Government -------------------------------
    ('government', 'medium', "Citizens can apply for the scheme through the official portal.",
     {'hindi': "नागरिक आधिकारिक पोर्टल के माध्यम से योजना के लिए आवेदन कर सकते हैं।",
      'bengali': "নাগরিকরা সরকারি পোর্টালের মাধ্যমে প্রকল্পের জন্য আবেদন করতে পারেন।",
      'marathi': "नागरिक अधिकृत पोर्टलद्वारे योजनेसाठी अर्ज करू शकतात.",
      'gujarati': "નાગરિકો સત્તાવાર પોર્ટલ દ્વારા યોજના માટે અરજી કરી શકે છે.",
      'tamil': "குடிமக்கள் அதிகாரப்பூர்வ இணையதளம் மூலம் திட்டத்திற்கு விண்ணப்பிக்கலாம்.",
      'telugu': "పౌరులు అధికారిక పోర్టల్ ద్వారా పథకానికి దరఖాస్తు చేసుకోవచ్చు.",
      'kannada': "ನಾಗರಿಕರು ಅಧಿಕೃತ ಪೋರ್ಟಲ್ ಮೂಲಕ ಯೋಜನೆಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಬಹುದು."}),
    ('government', 'simple', "The government announced a new policy for farmers.",
     {'hindi': "सरकार ने किसानों के लिए एक नई नीति की घोषणा की।",
      'bengali': "সরকার কৃষকদের জন্য একটি নতুন নীতি ঘোষণা করেছে।",
      'marathi': "सरकारने शेतकऱ्यांसाठी नवीन धोरण जाहीर केले.",
      'gujarati': "સરકારે ખેડૂતો માટે નવી નીતિ જાહેર કરી.",
      'tamil': "அரசு விவசாயிகளுக்காக புதிய கொள்கையை அறிவித்தது.",
      'telugu': "ప్రభుత్వం రైతుల కోసం కొత్త విధానాన్ని ప్రకటించింది.",
      'kannada': "ಸರ್ಕಾರವು ರೈತರಿಗಾಗಿ ಹೊಸ ನೀತಿಯನ್ನು ಘೋಷಿಸಿತು."}),
    ('government', 'medium', "Voters must carry a valid identity card to the polling booth.",
     {'hindi': "मतदाताओं को मतदान केंद्र पर वैध पहचान पत्र ले जाना चाहिए।",
      'bengali': "ভোটারদের ভোটকেন্দ্রে বৈধ পরিচয়পত্র নিয়ে যেতে হবে।",
      'marathi': "मतदारांनी मतदान केंद्रावर वैध ओळखपत्र नेले पाहिजे.",
      'gujarati': "મતદારોએ મતદાન મથક પર માન્ય ઓળખપત્ર લઈ જવું જોઈએ.",
      'tamil': "வாக்காளர்கள் வாக்குச்சாவடிக்கு செல்லுபடியாகும் அடையாள அட்டையை எடுத்துச் செல்ல வேண்டும்.",
      'telugu': "ఓటర్లు పోలింగ్ కేంద్రానికి చెల్లుబాటు అయ్యే గుర్తింపు కార్డు తీసుకెళ్లాలి.",
      'kannada': "ಮತದಾರರು ಮತಗಟ್ಟೆಗೆ ಮಾನ್ಯವಾದ ಗುರುತಿನ ಚೀಟಿಯನ್ನು ತೆಗೆದುಕೊಂಡು ಹೋಗಬೇಕು."}),

    # --------------------------- Legal ------------------------------------
    ('legal', 'simple', "The agreement will remain valid for one year.",
     {'hindi': "समझौता एक वर्ष तक वैध रहेगा।",
      'bengali': "চুক্তিটি এক বছরের জন্য বৈধ থাকবে।",
      'marathi': "करार एका वर्षासाठी वैध राहील.",
      'gujarati': "કરાર એક વર્ષ માટે માન્ય રહેશે.",
      'tamil': "ஒப்பந்தம் ஒரு ஆண்டுக்கு செல்லுபடியாகும்.",
      'telugu': "ఒప్పందం ఒక సంవత్సరం పాటు చెల్లుబాటు అవుతుంది.",
      'kannada': "ಒಪ್ಪಂದವು ಒಂದು ವರ್ಷದವರೆಗೆ ಮಾನ್ಯವಾಗಿರುತ್ತದೆ."}),
    ('legal', 'simple', "The court will hear the case next month.",
     {'hindi': "अदालत अगले महीने मामले की सुनवाई करेगी।",
      'bengali': "আদালত আগামী মাসে মামলার শুনানি করবে।",
      'marathi': "न्यायालय पुढील महिन्यात खटल्याची सुनावणी करेल.",
      'gujarati': "કોર્ટ આવતા મહિને કેસની સુનાવણી કરશે.",
      'tamil': "நீதிமன்றம் அடுத்த மாதம் வழக்கை விசாரிக்கும்.",
      'telugu': "కోర్టు వచ్చే నెలలో కేసును విచారిస్తుంది.",
      'kannada': "ನ್ಯಾಯಾಲಯವು ಮುಂದಿನ ತಿಂಗಳು ಪ್ರಕರಣವನ್ನು ವಿಚಾರಣೆ ಮಾಡುತ್ತದೆ."}),
    ('legal', 'complex', "Both parties must sign the contract before the deadline.",
     {'hindi': "दोनों पक्षों को समय सीमा से पहले अनुबंध पर हस्ताक्षर करना होगा।",
      'bengali': "উভয় পক্ষকে সময়সীমার আগে চুক্তিতে স্বাক্ষর করতে হবে।",
      'marathi': "दोन्ही पक्षांनी अंतिम मुदतीपूर्वी करारावर स्वाक्षरी करणे आवश्यक आहे.",
      'gujarati': "બંને પક્ષોએ સમયમર્યાદા પહેલા કરાર પર સહી કરવી પડશે.",
      'tamil': "இரு தரப்பும் காலக்கெடுவுக்குள் ஒப்பந்தத்தில் கையெழுத்திட வேண்டும்.",
      'telugu': "రెండు పక్షాలు గడువుకు ముందే ఒప్పందంపై సంతకం చేయాలి.",
      'kannada': "ಎರಡೂ ಪಕ್ಷಗಳು ಗಡುವಿನ ಮೊದಲು ಒಪ್ಪಂದಕ್ಕೆ ಸಹಿ ಹಾಕಬೇಕು."}),

    # --------------------------- Conversational ---------------------------
    ('conversational', 'simple', "Could you please send me the document tomorrow?",
     {'hindi': "क्या आप कल मुझे दस्तावेज़ भेज सकते हैं?",
      'bengali': "আপনি কি আগামীকাল আমাকে ডকুমেন্টটি পাঠাতে পারবেন?",
      'marathi': "कृपया तुम्ही उद्या मला दस्तऐवज पाठवू शकता का?",
      'gujarati': "શું તમે કાલે મને દસ્તાવેજ મોકલી શકશો?",
      'tamil': "நாளை எனக்கு ஆவணத்தை அனுப்ப முடியுமா?",
      'telugu': "రేపు మీరు నాకు డాక్యుమెంట్ పంపగలరా?",
      'kannada': "ನೀವು ನಾಳೆ ನನಗೆ ಡಾಕ್ಯುಮೆಂಟ್ ಕಳುಹಿಸಬಹುದೇ?"}),
    ('conversational', 'simple', "I will call you when I reach home.",
     {'hindi': "घर पहुँचते ही मैं आपको फोन करूँगा।",
      'bengali': "আমি বাড়ি পৌঁছালে আপনাকে ফোন করব।",
      'marathi': "मी घरी पोहोचताच तुम्हाला फोन करेन.",
      'gujarati': "હું ઘરે પહોંચતાં જ તમને ફોન કરીશ.",
      'tamil': "நான் வீட்டை அடைந்ததும் உங்களை அழைக்கிறேன்.",
      'telugu': "నేను ఇంటికి చేరుకున్న వెంటనే మిమ్మల్ని పిలుస్తాను.",
      'kannada': "ನಾನು ಮನೆ ತಲುಪಿದ ತಕ್ಷಣ ನಿಮಗೆ ಕರೆ ಮಾಡುತ್ತೇನೆ."}),
    ('conversational', 'colloquial', "Thanks for your help, see you soon.",
     {'hindi': "आपकी मदद के लिए धन्यवाद, जल्द मिलते हैं।",
      'bengali': "আপনার সাহায্যের জন্য ধন্যবাদ, শীঘ্রই দেখা হবে।",
      'marathi': "तुमच्या मदतीबद्दल धन्यवाद, लवकरच भेटू.",
      'gujarati': "તમારી મદદ બદલ આભાર, ટૂંક સમયમાં મળીશું.",
      'tamil': "உங்கள் உதவிக்கு நன்றி, விரைவில் சந்திப்போம்.",
      'telugu': "మీ సహాయానికి ధన్యవాదాలు, త్వరలో కలుద్దాం.",
      'kannada': "ನಿಮ್ಮ ಸಹಾಯಕ್ಕೆ ಧನ್ಯವಾದಗಳು, ಶೀಘ್ರದಲ್ಲೇ ಭೇಟಿಯಾಗೋಣ."}),

    # --------------------------- Difficult ---------------------------------
    ('difficult', 'simple', "I am going to college tomorrow.",
     {'hindi': "मैं कल कॉलेज जा रहा हूँ।",
      'bengali': "আমি আগামীকাল কলেজে যাচ্ছি।",
      'marathi': "मी उद्या कॉलेजला जात आहे.",
      'gujarati': "હું કાલે કૉલેજ જાઉં છું.",
      'tamil': "நான் நாளை கல்லூரிக்கு செல்கிறேன்.",
      'telugu': "నేను రేపు కాలేజీకి వెళ్తున్నాను.",
      'kannada': "ನಾನು ನಾಳೆ ಕಾಲೇಜಿಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ."}),
    ('difficult', 'medium', "The train to Mumbai leaves at six o'clock in the evening.",
     {'hindi': "मुंबई जाने वाली ट्रेन शाम छह बजे रवाना होती है।",
      'bengali': "মুম্বাই যাওয়া ট্রেনটি সন্ধ্যা ছয়টায় ছাড়ে।",
      'marathi': "मुंबईला जाणारी रेल्वे संध्याकाळी सहा वाजता सुटते.",
      'gujarati': "મુંબઈ જતી ટ્રેન સાંજે છ વાગ્યે ઉપડે છે.",
      'tamil': "மும்பை செல்லும் ரயில் மாலை ஆறு மணிக்கு புறப்படுகிறது.",
      'telugu': "ముంబై వెళ్లే రైలు సాయంత్రం ఆరు గంటలకు బయలుదేరుతుంది.",
      'kannada': "ಮುಂಬೈಗೆ ಹೋಗುವ ರೈಲು ಸಂಜೆ ಆರು ಗಂಟೆಗೆ ಹೊರಡುತ್ತದೆ."}),
    ('difficult', 'complex', "Although the weather was bad, the students decided to attend the event.",
     {'hindi': "हालाँकि मौसम खराब था, फिर भी छात्रों ने कार्यक्रम में शामिल होने का निर्णय लिया।",
      'bengali': "যদিও আবহাওয়া খারাপ ছিল, তবুও শিক্ষার্থীরা অনুষ্ঠানে যোগ দেওয়ার সিদ্ধান্ত নিয়েছে।",
      'marathi': "हवामान खराब असले तरी विद्यार्थ्यांनी कार्यक्रमाला उपस्थित राहण्याचा निर्णय घेतला.",
      'gujarati': "હવામાન ખરાબ હોવા છતાં, વિદ્યાર્થીઓએ કાર્યક્રમમાં હાજરી આપવાનું નક્કી કર્યું.",
      'tamil': "வானிலை மோசமாக இருந்தாலும், மாணவர்கள் நிகழ்ச்சியில் கலந்து கொள்ள முடிவு செய்தனர்.",
      'telugu': "వాతావరణం చెడ్డగా ఉన్నప్పటికీ, విద్యార్థులు కార్యక్రమానికి హాజరు కావాలని నిర్ణయించుకున్నారు.",
      'kannada': "ಹವಾಮಾನ ಕೆಟ್ಟದಾಗಿದ್ದರೂ, ವಿದ್ಯಾರ್ಥಿಗಳು ಕಾರ್ಯಕ್ರಮಕ್ಕೆ ಹಾಜರಾಗಲು ನಿರ್ಧರಿಸಿದರು."}),
    ('difficult', 'idiom', "Don't judge a book by its cover.",
     {'hindi': "किताब को उसके आवरण से मत आंकिए।",
      'bengali': "বইয়ের প্রচ্ছদ দেখে বইকে বিচার করবেন না।",
      'marathi': "पुस्तकाचा न्याय त्याच्या मुखपृष्ठावरून करू नका.",
      'gujarati': "પુસ્તકને તેના આવરણથી ન આંકો.",
      'tamil': "புத்தகத்தை அதன் அட்டையை வைத்து மதிப்பிடாதீர்கள்.",
      'telugu': "పుస్తకాన్ని దాని ముఖచిత్రం చూసి అంచనా వేయకండి.",
      'kannada': "ಪುಸ್ತಕವನ್ನು ಅದರ ಹೊದಿಕೆಯಿಂದ ನಿರ್ಣಯಿಸಬೇಡಿ."}),
    ('difficult', 'names', "Mr. Sharma bought a house in Pune last year.",
     {'hindi': "श्री शर्मा ने पिछले साल पुणे में एक घर खरीदा।",
      'bengali': "জনাব শর্মা গত বছর পুনেতে একটি বাড়ি কিনেছেন।",
      'marathi': "श्री शर्मा यांनी गेल्या वर्षी पुण्यात घर विकत घेतले.",
      'gujarati': "શ્રી શર્માએ ગયા વર્ષે પુણેમાં એક ઘર ખરીદ્યું.",
      'tamil': "திரு சர்மா கடந்த ஆண்டு புனேவில் ஒரு வீட்டை வாங்கினார்.",
      'telugu': "శ్రీ శర్మ గత సంవత్సరం పూణేలో ఒక ఇల్లు కొన్నారు.",
      'kannada': "ಶ್ರೀ ಶರ್ಮಾ ಕಳೆದ ವರ್ಷ ಪುಣೆಯಲ್ಲಿ ಮನೆ ಖರೀದಿಸಿದರು."}),
    ('difficult', 'numbers', "The total cost is 12,450 rupees including 8% tax.",
     {'hindi': "8% कर सहित कुल लागत 12,450 रुपये है।",
      'bengali': "8% করসহ মোট খরচ 12,450 টাকা।",
      'marathi': "8% करासह एकूण किंमत 12,450 रुपये आहे.",
      'gujarati': "8% કર સહિત કુલ કિંમત 12,450 રૂપિયા છે.",
      'tamil': "8% வரி உட்பட மொத்த செலவு 12,450 ரூபாய்.",
      'telugu': "8% పన్నుతో సహా మొత్తం ఖర్చు 12,450 రూపాయలు.",
      'kannada': "8% ತೆರಿಗೆ ಸೇರಿ ಒಟ್ಟು ವೆಚ್ಚ 12,450 ರೂಪಾಯಿ."}),
    ('difficult', 'dates', "The conference is scheduled for 15 August 2026.",
     {'hindi': "सम्मेलन 15 अगस्त 2026 को आयोजित होने वाला है।",
      'bengali': "সম্মেলনটি 15 আগস্ট 2026 তারিখে নির্ধারিত।",
      'marathi': "परिषद 15 ऑगस्ट 2026 रोजी होणार आहे.",
      'gujarati': "પરિષદ 15 ઓગસ્ટ 2026ના રોજ યોજાશે.",
      'tamil': "மாநாடு 15 ஆகஸ்ட் 2026 அன்று நடைபெறும்.",
      'telugu': "సమావేశం 15 ఆగస్టు 2026 న జరగాల్సి ఉంది.",
      'kannada': "ಸಮ್ಮೇಳನವು 15 ಆಗಸ್ಟ್ 2026 ರಂದು ನಿಗದಿಯಾಗಿದೆ."}),
    ('difficult', 'technical', "The neural network was trained on a parallel corpus of one million sentence pairs.",
     {'hindi': "तंत्रिका नेटवर्क को दस लाख वाक्य जोड़ों के समानांतर कोष पर प्रशिक्षित किया गया था।",
      'bengali': "নিউরাল নেটওয়ার্কটিকে দশ লক্ষ বাক্য জোড়ার সমান্তরাল কর্পাসে প্রশিক্ষণ দেওয়া হয়েছিল।",
      'marathi': "न्यूरल नेटवर्कला दहा लाख वाक्य जोड्यांच्या समांतर संग्रहावर प्रशिक्षण देण्यात आले.",
      'gujarati': "ન્યુરલ નેટવર્કને દસ લાખ વાક્ય જોડીના સમાંતર કોર્પસ પર તાલીમ આપવામાં આવી હતી.",
      'tamil': "நரம்பியல் வலையமைப்பு பத்து லட்சம் வாக்கிய இணைகள் கொண்ட இணைத் தொகுப்பில் பயிற்றுவிக்கப்பட்டது.",
      'telugu': "న్యూరల్ నెట్వర్క్కు పది లక్షల వాక్య జతల సమాంతర కార్పస్పై శిక్షణ ఇవ్వబడింది.",
      'kannada': "ನರ ಜಾಲಕ್ಕೆ ಹತ್ತು ಲಕ್ಷ ವಾಕ್ಯ ಜೋಡಿಗಳ ಸಮಾನಾಂತರ ಕಾರ್ಪಸ್ ಮೇಲೆ ತರಬೇತಿ ನೀಡಲಾಯಿತು."}),
    ('difficult', 'long', "The committee formed last year to study the impact of climate change on agriculture in the region has published its detailed report, which recommends several new measures to protect small farmers from unpredictable weather.",
     {'hindi': "जलवायु परिवर्तन के क्षेत्र में कृषि पर प्रभाव का अध्ययन करने के लिए पिछले साल गठित समिति ने अपनी विस्तृत रिपोर्ट प्रकाशित की है, जिसमें छोटे किसानों को अप्रत्याशित मौसम से बचाने के लिए कई नए उपायों की सिफारिश की गई है।",
      'bengali': "জলবায়ু পরিবর্তনের অঞ্চলের কৃষিতে প্রভাব অধ্যয়নের জন্য গত বছর গঠিত কমিটি তার বিস্তারিত প্রতিবেদন প্রকাশ করেছে, যেখানে ছোট কৃষকদের অপ্রত্যাশিত আবহাওয়া থেকে রক্ষা করার জন্য বেশ কয়েকটি নতুন ব্যবস্থার সুপারিশ করা হয়েছে।",
      'marathi': "प्रदेशातील शेतीवर हवामान बदलाच्या परिणामाचा अभ्यास करण्यासाठी गेल्या वर्षी स्थापन झालेल्या समितीने आपला तपशीलवार अहवाल प्रकाशित केला आहे, ज्यामध्ये लहान शेतकऱ्यांचे अनपेक्षित हवामानापासून संरक्षण करण्यासाठी अनेक नवीन उपायांची शिफारस केली आहे.",
      'gujarati': "આ પ્રદેશમાં ખેતી પર આબોહવા પરિવર્તનની અસરનો અભ્યાસ કરવા ગયા વર્ષે રચાયેલી સમિતિએ પોતાનો વિગતવાર અહેવાલ પ્રકાશિત કર્યો છે, જેમાં નાના ખેડૂતોને અણધારી હવામાનથી બચાવવા માટે ઘણા નવા પગલાંની ભલામણ કરવામાં આવી છે.",
      'tamil': "இப்பகுதியில் விவசாயத்தில் காலநிலை மாற்றத்தின் தாக்கத்தை ஆய்வு செய்ய கடந்த ஆண்டு அமைக்கப்பட்ட குழு தனது விரிவான அறிக்கையை வெளியிட்டுள்ளது, இது சிறு விவசாயிகளை எதிர்பாராத வானிலையிலிருந்து பாதுகாக்க பல புதிய நடவடிக்கைகளை பரிந்துரைக்கிறது.",
      'telugu': "ఈ ప్రాంతంలో వ్యవసాయంపై వాతావరణ మార్పుల ప్రభావాన్ని అధ్యయనం చేయడానికి గత సంవత్సరం ఏర్పాటైన కమిటీ తన వివరణాత్మక నివేదికను ప్రచురించింది, ఇది చిన్న రైతులను అనూహ్య వాతావరణం నుండి రక్షించడానికి అనేక కొత్త చర్యలను సిఫార్సు చేస్తుంది.",
      'kannada': "ಪ್ರದೇಶದ ಕೃಷಿಯ ಮೇಲೆ ಹವಾಮಾನ ಬದಲಾವಣೆಯ ಪರಿಣಾಮವನ್ನು ಅಧ್ಯಯನ ಮಾಡಲು ಕಳೆದ ವರ್ಷ ರಚಿಸಲಾದ ಸಮಿತಿಯು ತನ್ನ ವಿವರವಾದ ವರದಿಯನ್ನು ಪ್ರಕಟಿಸಿದೆ, ಇದು ಸಣ್ಣ ರೈತರನ್ನು ಅನಿರೀಕ್ಷಿತ ಹವಾಮಾನದಿಂದ ರಕ್ಷಿಸಲು ಹಲವಾರು ಹೊಸ ಕ್ರಮಗಳನ್ನು ಶಿಫಾರಸು ಮಾಡುತ್ತದೆ."}),
    ('difficult', 'colloquial', "This movie is really awesome, you must watch it tonight!",
     {'hindi': "यह फिल्म सच में कमाल की है, तुम्हें इसे आज रात ज़रूर देखना चाहिए!",
      'bengali': "ছবিটা সত্যিই দারুণ, তোমাকে আজ রাতে এটা অবশ্যই দেখতে হবে!",
      'marathi': "हा चित्रपट खरोखर छान आहे, तुम्ही तो आज रात्री नक्की पाहिला पाहिजे!",
      'gujarati': "આ ફિલ્મ ખરેખર અદ્ભુત છે, તમારે તે આજે રાત્રે જરૂર જોવી જોઈએ!",
      'tamil': "இந்தப் படம் உண்மையிலேயே அருமை, இன்று இரவே நீங்கள் அதைப் பார்க்க வேண்டும்!",
      'telugu': "ఈ సినిమా నిజంగా చాలా బాగుంది, ఈ రాత్రే మీరు దీన్ని తప్పకుండా చూడాలి!",
      'kannada': "ಈ ಚಿತ್ರ ನಿಜವಾಗಿಯೂ ಅದ್ಭುತವಾಗಿದೆ, ನೀವು ಅದನ್ನು ಇಂದು ರಾತ್ರಿಯೇ ನೋಡಲೇಬೇಕು!"}),
]


###############################################################################
# Code-mixed benchmark (Indic script + English words -> English reference)
###############################################################################

CODE_MIXED_ITEMS = [
    # 25 sentences: 4 each for hi/mr/bn/gu, 3 each for ta/te/kn.
    # --------------------------- Hindi ----------------------------------
    {'source_lang': 'hindi', 'source': "आज मुझे college में एक important presentation देना है।",
     'reference': "Today I have to give an important presentation at college."},
    {'source_lang': 'hindi', 'source': "मुझे आज office जाना है।",
     'reference': "I have to go to the office today."},
    {'source_lang': 'hindi', 'source': "इस project की deadline कल है, इसलिए हमें जल्दी finish करना होगा।",
     'reference': "The deadline for this project is tomorrow, so we have to finish it soon."},
    {'source_lang': 'hindi', 'source': "मैं weekend पर अपनी family के साथ mall जा रहा हूँ।",
     'reference': "I am going to the mall with my family this weekend."},

    # --------------------------- Marathi --------------------------------
    {'source_lang': 'marathi', 'source': "मला उद्या meeting ला जायचं आहे.",
     'reference': "I have to go to a meeting tomorrow."},
    {'source_lang': 'marathi', 'source': "हा project आपल्याला Friday पर्यंत complete करायचा आहे.",
     'reference': "We have to complete this project by Friday."},
    {'source_lang': 'marathi', 'source': "माझ्या brother ला नवीन job मिळाली आहे.",
     'reference': "My brother got a new job."},
    {'source_lang': 'marathi', 'source': "आपण उद्या video call वर discuss करूया.",
     'reference': "Let us discuss over a video call tomorrow."},

    # --------------------------- Bengali --------------------------------
    {'source_lang': 'bengali', 'source': "আমি কাল doctor-এর কাছে যাব।",
     'reference': "I will go to the doctor tomorrow."},
    {'source_lang': 'bengali', 'source': "এই project-টা আগামীকাল submit করতে হবে।",
     'reference': "This project has to be submitted tomorrow."},
    {'source_lang': 'bengali', 'source': "আমার friends movie দেখতে যাচ্ছে।",
     'reference': "My friends are going to watch a movie."},
    {'source_lang': 'bengali', 'source': "office-এ নতুন policy চালু হয়েছে।",
     'reference': "A new policy has been launched at the office."},

    # --------------------------- Gujarati -------------------------------
    {'source_lang': 'gujarati', 'source': "મારે આજે bank જવું છે.",
     'reference': "I have to go to the bank today."},
    {'source_lang': 'gujarati', 'source': "આ project ની deadline કાલે છે.",
     'reference': "The deadline for this project is tomorrow."},
    {'source_lang': 'gujarati', 'source': "મારા friend ને નવી job મળી છે.",
     'reference': "My friend got a new job."},
    {'source_lang': 'gujarati', 'source': "આપણે ઉદ્યા video call પર discuss કરીશું.",
     'reference': "We will discuss over a video call tomorrow."},

    # --------------------------- Tamil ----------------------------------
    {'source_lang': 'tamil', 'source': "நான் இன்று office-க்கு போகிறேன்.",
     'reference': "I am going to the office today."},
    {'source_lang': 'tamil', 'source': "இந்த project-ஐ நாளைக்கு submit பண்ணணும்.",
     'reference': "This project has to be submitted tomorrow."},
    {'source_lang': 'tamil', 'source': "நான் weekend-ல family-உடன் mall-க்கு போறேன்.",
     'reference': "I am going to the mall with my family this weekend."},

    # --------------------------- Telugu ---------------------------------
    {'source_lang': 'telugu', 'source': "నేను రేపు meeting కి వెళ్తున్నాను.",
     'reference': "I am going to a meeting tomorrow."},
    {'source_lang': 'telugu', 'source': "ఈ project ని Friday లోపల complete చేయాలి.",
     'reference': "This project has to be completed by Friday."},
    {'source_lang': 'telugu', 'source': "నాకు ఈ job లో salary చాలా బాగుంది.",
     'reference': "The salary in this job is very good for me."},

    # --------------------------- Kannada --------------------------------
    {'source_lang': 'kannada', 'source': "ನಾನು ನಾಳೆ meeting ಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ.",
     'reference': "I am going to a meeting tomorrow."},
    {'source_lang': 'kannada', 'source': "ಈ project ಅನ್ನು Friday ಒಳಗೆ complete ಮಾಡಬೇಕು.",
     'reference': "This project has to be completed by Friday."},
    {'source_lang': 'kannada', 'source': "ನನ್ನ friend ಗೆ ಹೊಸ job ಸಿಕ್ಕಿದೆ.",
     'reference': "My friend got a new job."},
]


###############################################################################
# Backward-compatible aggregate views (used by routes + run_evaluation.py)
###############################################################################

ENGLISH_SENTENCES: List[str] = [item[2] for item in CURATED_ITEMS]

DOMAIN_LABELS: List[str] = [item[0] for item in CURATED_ITEMS]

CATEGORY_LABELS: List[str] = [item[1] for item in CURATED_ITEMS]

REFERENCES: Dict[str, List[str]] = {
    lang: [item[3].get(lang, '') for item in CURATED_ITEMS]
    for lang in CURATED_LANGS
}


def get_benchmark_dataset(source_lang: str = 'english', target_langs: list = None):
    """
    Get benchmark dataset for specified language pair(s).

    Args:
        source_lang: Source language (default: 'english')
        target_langs: List of target language codes. If None, all available.

    Returns:
        dict with sources, references dict, domains, categories, metadata
    """
    if target_langs is None:
        target_langs = list(REFERENCES.keys())

    available = [l for l in target_langs if l in REFERENCES]
    if not available:
        return {'error': f'No datasets available for: {target_langs}'}

    return {
        'sources': ENGLISH_SENTENCES,
        'references': {lang: REFERENCES[lang] for lang in available},
        'domains': DOMAIN_LABELS,
        'categories': CATEGORY_LABELS,
        'metadata': {
            'source_lang': source_lang,
            'target_langs': available,
            'num_sentences': len(ENGLISH_SENTENCES),
            'num_domains': len(set(DOMAIN_LABELS)),
            'languages': {lang: LANGUAGE_META[lang] for lang in available},
            'reference_note': (
                'Curated references are machine-generated and lightly edited. '
                'Verify with native speakers before publication.'
            ),
        },
    }


def get_quality_dataset(target_langs: list = None):
    """
    Alias for get_benchmark_dataset — the curated quality benchmark.
    Included so the name matches the benchmark report sections.
    """
    return get_benchmark_dataset(target_langs=target_langs)


def get_code_mixed_dataset():
    """
    Get the code-mixed translation benchmark.

    Each item: {source_lang, source, reference}. Target language is always
    'english' (the code-mixed sentence is translated into clean English).
    """
    return {
        'items': CODE_MIXED_ITEMS,
        'target_lang': 'english',
        'metadata': {
            'num_sentences': len(CODE_MIXED_ITEMS),
            'languages': sorted({i['source_lang'] for i in CODE_MIXED_ITEMS}),
        },
    }


def get_domain_split():
    """
    Get dataset grouped by domain for domain-wise evaluation.

    Returns:
        dict: domain -> [{'sentence': ..., 'references': {lang: ...}}]
    """
    domains = {}
    for i, domain in enumerate(DOMAIN_LABELS):
        if domain not in domains:
            domains[domain] = {'sources': [], 'references': {}, 'categories': []}
        domains[domain]['sources'].append(ENGLISH_SENTENCES[i])
        domains[domain]['categories'].append(CATEGORY_LABELS[i])
        for lang in REFERENCES:
            if lang not in domains[domain]['references']:
                domains[domain]['references'][lang] = []
            domains[domain]['references'][lang].append(REFERENCES[lang][i])
    return domains


def get_category_split():
    """
    Get dataset grouped by difficulty category (simple/medium/complex/...).

    Returns:
        dict: category -> {'sources': [...], 'references': {lang: [...]}}
    """
    categories = {}
    for i, category in enumerate(CATEGORY_LABELS):
        if category not in categories:
            categories[category] = {'sources': [], 'references': {}, 'domains': []}
        categories[category]['sources'].append(ENGLISH_SENTENCES[i])
        categories[category]['domains'].append(DOMAIN_LABELS[i])
        for lang in REFERENCES:
            if lang not in categories[category]['references']:
                categories[category]['references'][lang] = []
            categories[category]['references'][lang].append(REFERENCES[lang][i])
    return categories


###############################################################################
# FLORES-200 loader (optional, research-grade parallel corpus)
###############################################################################

FLORES_REPO = 'facebook/flores'


def _read_parquet_text(path):
    """Read a FLORES per-language parquet and return the list of sentences.

    Tries the standard column names ('sentence', 'text'), then falls back to
    the first non-metadata string column.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    cols = table.column_names
    for candidate in ('sentence', 'text', 'translation', 'src_text'):
        if candidate in cols:
            return [v.as_py() for v in table.column(candidate)]
    # Fallback: first column that is not id/URL metadata
    for col in cols:
        if col.lower() not in ('id', 'url', 'index', 'split', 'lang', 'language'):
            return [v.as_py() for v in table.column(col)]
    return []


def load_flores_dataset(
    target_langs: Optional[List[str]] = None,
    split: str = 'devtest',
    max_sentences: Optional[int] = None,
    source_lang: str = 'english',
    local_dir: Optional[str] = None,
):
    """
    Load FLORES-200 parallel sentences (research-grade references).

    LOCAL FILES: pass ``local_dir`` (or set FLORES_DIR) to read the plain-text
    release instead of the Hub (layout of flores200_dataset.tar.gz:
    ``{local_dir}/{split}/{code}.{split}``, one sentence per line).

    FLORES-200 (facebook/flores) contains 1012 devtest sentences per
    language in 200 languages. This loader downloads the per-language
    parquet files directly from the Hub and zips them into the same
    structure as get_benchmark_dataset (sources + references per lang).

    AUTH: facebook/flores is a GATED dataset. You must authenticate first:

        huggingface-cli login        # or set the HF_TOKEN env var

    Requires: pip install huggingface_hub pyarrow  (internet on first use)

    Args:
        target_langs: Languages to load (default: all supported except source).
        split: 'dev', 'devtest' or 'test' (default 'devtest').
        max_sentences: Cap the number of sentences (default: all 1012).
        source_lang: Source language name (default 'english').

    Returns:
        dict with sources, references, metadata — or {'error': ...} on failure.
    """
    if target_langs is None:
        # Default: every supported language except the source (no x->x pairs)
        target_langs = [l for l in FLORES_CODES if l != source_lang]

    src_code = FLORES_CODES.get(source_lang)
    if not src_code:
        return {'error': f'Unknown source language: {source_lang}'}

    target_codes = {}
    for lang in target_langs:
        code = FLORES_CODES.get(lang)
        if code:
            target_codes[lang] = code

    if not target_codes:
        return {'error': f'No supported target languages in: {target_langs}'}

    local_dir = local_dir or os.getenv('FLORES_DIR')
    if local_dir:
        return _load_flores_local(local_dir, split, src_code, source_lang,
                                  target_codes, max_sentences)

    def _fetch(code):
        """Download one per-language devtest parquet via the Hub API."""
        filename = f'data/language/{code}/{split}-00000-of-00001.parquet'
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=FLORES_REPO,
                filename=filename,
                repo_type='dataset',
            )
        except Exception as e:
            raise RuntimeError(
                f'Could not fetch {filename} from {FLORES_REPO}: {e}'
            ) from e

    try:
        src_path = _fetch(src_code)
        sources = _read_parquet_text(src_path)
    except RuntimeError as e:
        return {
            'error': (
                f'{e}\n\nNOTE: facebook/flores is gated. Authenticate with '
                '`huggingface-cli login` (or set the HF_TOKEN env var) and try '
                'again.'
            ),
        }
    except Exception as e:
        return {'error': f'Failed to load FLORES-200: {e}'}

    references = {}
    failed = []
    for lang, code in target_codes.items():
        try:
            path = _fetch(code)
            references[lang] = _read_parquet_text(path)
        except Exception:
            failed.append(lang)
            references[lang] = []

    if max_sentences:
        sources = sources[:max_sentences]
        references = {k: v[:max_sentences] for k, v in references.items()}

    return {
        'sources': sources,
        'references': references,
        'metadata': {
            'source_lang': source_lang,
            'source': f'FLORES-200 ({FLORES_REPO}, split={split})',
            'split': split,
            'num_sentences': len(sources),
            'target_langs': list(references.keys()),
            'languages': {lang: LANGUAGE_META.get(lang, {}) for lang in references},
            'failed_languages': failed,
            'reference_note': 'FLORES-200 is the standard research-grade parallel corpus (NLLB/FLORES benchmark).',
        },
    }


def _load_flores_local(local_dir, split, src_code, source_lang, target_codes,
                       max_sentences):
    """Read FLORES-200 plain-text files ({dir}/{split}/{code}.{split})."""
    def _read(code):
        path = os.path.join(local_dir, split, f'{code}.{split}')
        with open(path, 'r', encoding='utf-8') as f:
            return [line.rstrip('\n') for line in f]

    try:
        sources = _read(src_code)
    except OSError as e:
        return {'error': f'Could not read FLORES source file: {e}'}
    references, failed = {}, []
    for lang, code in target_codes.items():
        try:
            references[lang] = _read(code)
        except OSError:
            failed.append(lang)
            references[lang] = []
        if references[lang] and len(references[lang]) != len(sources):
            return {'error': f'FLORES {lang} has {len(references[lang])} lines, '
                             f'source has {len(sources)}'}
    if max_sentences:
        sources = sources[:max_sentences]
        references = {k: v[:max_sentences] for k, v in references.items()}
    return {
        'sources': sources,
        'references': references,
        'metadata': {
            'source_lang': source_lang,
            'source': f'FLORES-200 (local: {local_dir}, split={split})',
            'split': split,
            'num_sentences': len(sources),
            'target_langs': list(references.keys()),
            'languages': {lang: LANGUAGE_META.get(lang, {}) for lang in references},
            'failed_languages': failed,
            'reference_note': 'FLORES-200 is the standard research-grade parallel corpus (NLLB/FLORES benchmark).',
        },
    }
