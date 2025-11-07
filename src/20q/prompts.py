## SINGLE-QUESTION PROMPTS 
SQ_PREDICTOR_SYSTEM_PROMPT = """Your goal is to predict the animal that the other player is thinking with the fewest number of questions. Your goal is to suggest a single at each iteration. You may ask any question you like. You can only make a single prediction. The animal must be one of the following: {class_names}. Be as precise and as direct as possible."""

SQ_PREDICTOR_PROMPT = """Given the information you have gathered, make an random prediction of what you think the animal is. Provide your reasoning first, then make your guess in the format 'The animal is:' (e.g. The animal is: dog). Do not provide any additional information. Your response should all fit in a single paragraph. Make sure your prediction is one of the classes: {class_names}."""

SQ_QUERY_ANSWERER_FIRST_PROMPT = """You may now suggest a single question."""

SQ_QUERIER_PROMPT = """Now suggest one question."""

SQ_QUERY_ANSWERER_SYSTEM_PROMPT = """You are an expert on {label}. Based on the question provided, answer truthfully about the question. Do not directly tell the other player what you are thinking. Be as precise and as direct as possible, and answer in complete sentence. For example, if the question is "Does the animal have a tail?", you can answer "The animal has a tail" without saying yes or no. Do not say the name of the animal in your answer."""

## MULTI-QUESTION PROMPTS
QUERIER_SYSTEM_PROMPT = """Your goal is to predict the animal that the other player is thinking with the fewest number of questions. Your goal is to suggest {n_queries_per_step} at each iteration. You may ask any question you like. You can only make a single prediction. The animal must be one of the following: {class_names}. Be as precise and as direct as possible. Return the question in this format:

{{"questions": ["QUESTION_1", "QUESTION_2", "QUESTION_3"]}}
"""

QUERIER_SYSTEM_PROMPT_DIRECT = """Your goal is to predict the animal that the other player is thinking with the fewest number of questions. Your goal is to suggest {n_queries_per_step} at each iteration. You may ask any question you like. You can only make a single prediction. The animal must be one of the following: {class_names}. Be as precise and as direct as possible. DO NOT ask directly what is the animal? Your question also cannot contain the class name, hence you cannot ask questions such as "Is it a giraffe?". DO NOT repeat the same question twice. Return the question in this format:

{{"questions": ["QUESTION_1", "QUESTION_2", "QUESTION_3"]}}
"""

# PREDICTOR_SYSTEM_PROMPT = """Your goal is to predict the animal that the other player is thinking with the fewest number of questions. Given the information you have gathered, you can asked to make a single prediction. The animal must be one of the following: {class_names}. Be as precise and as direct as possible."""

PREDICTOR_SYSTEM_PROMPT = "You are an expert on animals. Your goal is to predict the animal given the information you have gathered. The animal must be one of the following: {class_names}. Be as precise and as direct as possible. You may provide your reasoning first before making a prediction. End your response by making a guess and saying 'The animal is: X' (e.g. The animal is: dog) at the end of your reasoning, where X is your guess. Do not provide any additional information. Your response should all fit in a single paragraph. If you are not provided any information, make a random guess."
    
PREDICTOR_PROMPT = """Given the information you have gathered, make an intermediate SINGLE prediction of what you think the animal is. First make your guess in the format 'The ansmal is: X' (e.g. The animal is: dog), where X is your guess, then provide your reasoning. Do not provide any additional information. Your response should all fit in a single paragraph. Make sure your prediction is one of the classes: {class_names}."""

QUERY_ANSWERER_FIRST_PROMPT = """You may now suggest the {n_queries_per_step} questions."""

QUERIER_PROMPT = """{history}\nNow suggest {n_queries_per_step} questions.\nReturn the question in this format:\n{{"questions": ["QUESTION_1", "QUESTION_2", "QUESTION_3"]}}"""

QUERY_ANSWERER_SYSTEM_PROMPT = """You are an expert on {label}. Based on the question provided, answer truthfully about the question. Do not directly tell the other player what you are thinking. Be as precise and as direct as possible, and answer in complete sentence. For example, if the question is "Does the animal have a tail?", you can answer "The animal has a tail." without saying yes or no. Do not say the name of the animal in your answer."""

QUERY_ANSWERER_SYSTEM_PROMPT_YESNO = """You are an expert on {label}. Based on the question provided, answer truthfully about the question. Do not directly tell the other player what you are thinking. Be as precise and as direct as possible, and answer with a single word. For example, if the question is "Does the animal have a tail?", you can answer "Yes." or "No.". Do not say the name of the animal in your answer. If you don't know the answer, make a guess. Do not answer anything other than "Yes." or "No."."""

## FIXED QUERYSET PROMPTS
QUERIER_SYSTEM_PROMPT_CLOSED_DIRECT = """Your goal is to predict the animal that the other player is thinking with the fewest number of questions. Your goal is to suggest {n_queries_per_step} at each iteration. You may ask any question you like. You can only make a single prediction. The animal must be one of the following: {class_names}.
"""

QUERIER_PROMPT_CLOSED = """{history}\nNow suggest 1 question from the following list: {query_bank}.\nReturn the question in this format:\n{{"questions": ["QUESTION_1"]}}"""

## ONE TOKEN PROMPTS
PREDICTOR_PROMPT_USER = """Given the information you have gathered, make an intermediate prediction of what you think the animal is. Respond in the format 'The animal is:' (e.g. The animal is: dog). Do not provide any additional information. Your response should all fit in a single paragraph. Make sure your prediction is one of the classes: {class_names}."""

PREDICTOR_PROMPT_ASSISTANT = """The animal is:"""