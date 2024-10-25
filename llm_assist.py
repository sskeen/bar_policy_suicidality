
import time
import openai

api_key = 'OPENAI_API_KEY'
client = openai.OpenAI(api_key = api_key)

def annotate_post_per_tag(text, prompts):
    """
    Applies annotation decisions, based on multiple prompts, to a given text; provides rationale and explanation.
    Parameters:
    - text: The text to annotate.
    - prompts: A list of prompts to apply to the text.

    Returns:
    - result: The combined result from all prompts.
    """
    try:

        # concatenate prompts

        prompt_content = ' '.join(prompts)

        response = client.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.2,
            messages = [
                {
                    'role': 'system',
                    'content': prompt_content
                },
                {
                    'role': 'user',
                    'content': text
                }
            ]
        )

        # collect results

        result = ' '
        for choice in response.choices:
            result += choice.message.content

        print(f'{text}: {result}')
        return result
    except Exception as e:
        print(f'Exception: {e}')
        return 'error'

def annotate_dataframe_per_tag(df, prompts_per_tag):
    """
    Applies annotate_post_per_tag for multiple tags to each row in dataframe 'd'.

    Parameters:
    - df: The dataframe containing texts to annotate.
    - prompts_per_tag: A dictionary with tag names as keys and a list of prompts as values.

    Returns:
    - df: The updated dataframe with annotation results.
    """
    for index, row in df.iterrows():
        for tag, prompts in prompts_per_tag.items():
            result = annotate_post_per_tag(row['text'], prompts)
            if result == 'error':
                continue

            # extract rationale, chain of thought ("explanation")

            rationale, explanation = None, None
            if f'{tag}_1' in result:
                tag_value = 1
                rationale = result.split(f'{tag}_rationale:')[1].split(f'strained {tag}:')[0].strip() if f'{tag}_rationale:' in result else None

            # excise {tag}_explanation and subsequent text from rationale

                if rationale and f'{tag}_explanation:' in rationale:
                    rationale = rationale.split(f'{tag}_explanation:')[0].strip()

                #if f'{tag}_explanation:' in rationale:
                #    rationale = rationale.split(f'{tag}_explanation:')[0].strip()

                explanation = result.split(f'{tag}_explanation:')[1].strip() if f'{tag}_explanation:' in result else None
            else:
                tag_value = 0

            # results to df

            df.at[index, f'{tag}_gpt'] = tag_value
            df.at[index, f'{tag}_rtnl_gpt'] = rationale
            df.at[index, f'{tag}_expl_gpt'] = explanation

            # impose delay between API calls

            time.sleep(1)

    return df
