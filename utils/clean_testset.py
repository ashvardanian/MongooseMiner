from io import StringIO
import tokenize
import pandas as pd
from pprint import pprint


def clean_dataset(parquet_path: str, out_path: str):
    df = read_parquet(parquet_path)
    df["func_code_string"] = df["func_code_string"].apply(remove_comments_and_docstrings)
    cleaned_df = df[df["func_code_string"] != ""]
    cleaned_df.to_parquet(out_path, engine='fastparquet')
    print("Done!")

def read_parquet(parquet_path: str):
    df = pd.read_parquet(parquet_path, engine='fastparquet')
    return df

def remove_comments_and_docstrings(source):
    try:
        return _remove_comments_and_docstrings(source)
    except:
        return source

def _remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    tokens = tokenize.generate_tokens(io_obj.readline)
    for tok in tokens:
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
        # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
                    # For example:
                    # def foo():
                    #     "The spaces before this docstring are tokenize.INDENT"
                    #     test = [
                    #         "The spaces before this string do not get a token"
                    #     ]
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    pprint(out)
    pprint("-------------------------------------------------")
    return out


if __name__ == "__main__":
    PARQUET_PATH = "/home/michael/MongooseMiner/data/test.parquet"
    OUT_PATH = "/home/michael/MongooseMiner/data/test_cleaned.parquet"
    
    clean_dataset(PARQUET_PATH, OUT_PATH)