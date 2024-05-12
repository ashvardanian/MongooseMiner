import gradio as gr
import random
import time
import os

from groq import Groq

from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def user(user_message, history):
    return "", history + [[user_message, None]]


def call_service(query):
    docstring = """
    Return matrix rank of array using SVD method

    Rank of the array is the number of singular values of the array that are
    greater than `tol`.

    .. versionchanged:: 1.14
       Can now operate on stacks of matrices

    Parameters
    ----------
    A : {(M,), (..., M, N)} array_like
        Input vector or stack of matrices.
    tol : (...) array_like, float, optional
        Threshold below which SVD values are considered zero. If `tol` is
        None, and ``S`` is an array with singular values for `M`, and
        ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
        set to ``S.max() * max(M, N) * eps``.

        .. versionchanged:: 1.14
           Broadcasted against the stack of matrices
    hermitian : bool, optional
        If True, `A` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.

        .. versionadded:: 1.14
    rtol : (...) array_like, float, optional
        Parameter for the relative tolerance component. Only ``tol`` or
        ``rtol`` can be set at a time. Defaults to ``max(M, N) * eps``.

        .. versionadded:: 2.0.0

    Returns
    -------
    rank : (...) array_like
        Rank of A.

    Notes
    -----
    The default threshold to detect rank deficiency is a test on the magnitude
    of the singular values of `A`.  By default, we identify singular values
    less than ``S.max() * max(M, N) * eps`` as indicating rank deficiency
    (with the symbols defined above). This is the algorithm MATLAB uses [1].
    It also appears in *Numerical recipes* in the discussion of SVD solutions
    for linear least squares [2].

    This default threshold is designed to detect rank deficiency accounting
    for the numerical errors of the SVD computation. Imagine that there
    is a column in `A` that is an exact (in floating point) linear combination
    of other columns in `A`. Computing the SVD on `A` will not produce
    a singular value exactly equal to 0 in general: any difference of
    the smallest SVD value from 0 will be caused by numerical imprecision
    in the calculation of the SVD. Our threshold for small SVD values takes
    this numerical imprecision into account, and the default threshold will
    detect such numerical rank deficiency. The threshold may declare a matrix
    `A` rank deficient even if the linear combination of some columns of `A`
    is not exactly equal to another column of `A` but only numerically very
    close to another column of `A`.

    We chose our default threshold because it is in wide use. Other thresholds
    are possible.  For example, elsewhere in the 2007 edition of *Numerical
    recipes* there is an alternative threshold of ``S.max() *
    np.finfo(A.dtype).eps / 2. * np.sqrt(m + n + 1.)``. The authors describe
    this threshold as being based on "expected roundoff error" (p 71).

    The thresholds above deal with floating point roundoff error in the
    calculation of the SVD.  However, you may have more information about
    the sources of error in `A` that would make you consider other tolerance
    values to detect *effective* rank deficiency. The most useful measure
    of the tolerance depends on the operations you intend to use on your
    matrix. For example, if your data come from uncertain measurements with
    uncertainties greater than floating point epsilon, choosing a tolerance
    near that uncertainty may be preferable. The tolerance may be absolute
    if the uncertainties are absolute rather than relative.

    References
    ----------
    .. [1] MATLAB reference documentation, "Rank"
           https://www.mathworks.com/help/techdoc/ref/rank.html
    .. [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
           "Numerical Recipes (3rd edition)", Cambridge University Press, 2007,
           page 795.

    Examples
    --------
    >>> from numpy.linalg import matrix_rank
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    3
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> matrix_rank(np.zeros((4,)))
    0
    """

    end = f"""
             You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
             Question: {query}
             Context: {docstring}
             Answer: """
    return end


def generate_rag(history):
    history[-1][1] = ""
    stream = client.chat.completions.create(
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": call_service(history[-1][0])
            },
        ],
        stream=True,
        model="llama3-8b-8192",
        max_tokens=1024,
        temperature=0
    )

    for chunk in stream:
        if chunk.choices[0].delta.content != None:
            history[-1][1] += chunk.choices[0].delta.content
            yield history
        else:
            return


def generate_llama3(history):
    history[-1][1] = ""
    stream = client.chat.completions.create(
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": history[-1][0],
            }
        ],
        stream=True,
        model="llama3-8b-8192",
        max_tokens=1024,
        temperature=0
    )

    for chunk in stream:
        if chunk.choices[0].delta.content != None:
            history[-1][1] += chunk.choices[0].delta.content
            yield history
        else:
            return


with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            gr.Markdown("# Mongoose Miner Search Demo")
            gr.Markdown(
                "Augmenting LLM code generation with function-level search across all of PyPi.")

    with gr.Row():
        chatbot = gr.Chatbot(height="35rem", label="Llama3 unaugmented")
        chatbot2 = gr.Chatbot(
            height="35rem", label="Llama3 with MongooseMiner Search")
    msg = gr.Textbox()

    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_llama3, chatbot, chatbot
    )
    msg.submit(user, [msg, chatbot2], [msg, chatbot2], queue=False).then(
        generate_rag, chatbot2, chatbot2
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    clear.click(lambda: None, None, chatbot2, queue=False)


demo.queue()
demo.launch()
