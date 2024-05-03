from groq import Groq
def get_response():
    client = Groq()
    question = input('')
    response = client.chat.completions.create(
        model = 'llama3-8b-8192',
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant. Make sure not to add conclusion."
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )
    return question, response.choices[0].message.content

from IPython.display import Markdown, display

question, response = get_response()
display(Markdown(f'**Prompt:** {question}\n\n**Response:** {response}'))
