import streamlit as st
import jsonlines
import json
from gui_helper import *


class Paper:
    def __init__(self, paper_id, metadata, title, abstract, pred_labels_truncated, pred_labels):
        self.paper_id = paper_id
        self.metadata = metadata
        self.title = title
        self.abstract = abstract
        self.pred_labels_truncated = pred_labels_truncated
        self.pred_labels = pred_labels


papers = dict()
with jsonlines.open('../data/abstracts-csfcube-preds.jsonl') as reader:
    for obj in reader:
        papers[obj['paper_id']] = Paper(obj['paper_id'], obj['metadata'], obj['title'],
                                        obj['abstract'], obj['pred_labels_truncated'], obj['pred_labels'])

f = open('../data/test-pid2anns-csfcube-background.json')
q_background = json.load(f)
q_background_ids = list(q_background.keys())
q_background_ids.remove('1791179')
q_background_ids.insert(1, '1791179')

f = open('../data/test-pid2anns-csfcube-method.json')
q_method = json.load(f)
q_method_ids = list(q_method.keys())
q_method_ids.remove('10010426')
q_method_ids.insert(0, '10010426')

f = open('../data/test-pid2anns-csfcube-result.json')
q_result = json.load(f)
q_result_ids = list(q_result.keys())
q_result_ids.remove('3264891')
q_result_ids.insert(0, '3264891')

all_ids = list(papers.keys())

arr_background = []
for keys in q_background_ids:
    arr_background.append("[" + keys + "] " + papers[keys].title)
arr_method = []
for keys in q_method_ids:
    arr_method.append("[" + keys + "] " + papers[keys].title)
arr_result = []
for keys in q_result_ids:
    arr_result.append("[" + keys + "] " + papers[keys].title)

st.set_page_config(
    page_title="SciSearch",
    page_icon="🎈",
    layout="wide"
)


def _max_width_():
    max_width_str = f"max-width: 2400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

st.title("SciSearch - Query Research Papers More Efficiently")

with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """
        - *SciSearch* is a web application that allows users to query papers from the scientific literature. We aim to help the scientific community to find papers that are relevant to their research goals.
        - Currently this is a demo version that works for queries that are already present in CSFCube.
        - To expand on this, with some more effort we can make this a general purpose web application that can be used for any query paper.
        """
    )
    st.markdown("")

my_query = dict()
my_query["Background"] = arr_background
my_query["Method"] = arr_method
my_query["Result"] = arr_result

st.subheader("Facet Selection")

cc1, cc2 = st.columns([1, 4])

with cc1:
    st.session_state.facet = st.selectbox(
        "Choose the facet for which you want to query",
        (
            "Background",
            "Method",
            "Result"
        )
    )

st.session_state.query_selection = my_query[st.session_state.facet]
st.session_state.query_text = my_query[st.session_state.facet]


st.subheader("Model and Query Selection")
with st.form(key="my_form2"):
    ce, c1, ce, c2, ce, c3, ce = st.columns(
        [0.07, 1.5, 0.07, 1.5, 0.07, 5, 0.07])
    with c1:

        st.session_state.model = st.radio(
            "Choose the model",
            (
                "bert_nli",
                "bert_pp",
                "scibert_cased",
                "scibert_uncased",
                "specter",
                "susimcse",
                "unsimcse"
            )
            # index = (2 if st.session_state.facet == "Background" else 4)
        )

    st.markdown("")

    with c2:

        st.session_state.loss_func = st.radio(
            "Choose the loss function",
            (
                "Cross-Entropy-Loss",
                "Kullback-Leibler-Divergence-Loss"
            )
            # index = 1
        )

    st.markdown("")

    with c3:
        st.session_state.query_option = st.selectbox(
            "Choose the query",
            st.session_state.query_selection)

        st.session_state.top_N = st.slider(
            "Number of results",
            min_value=1,
            max_value=50,
            value=10,
            help="You can choose the number of papers to display. Between 1 and 50, default number is 10.",
        )
        submit_button = st.form_submit_button(label="Find Papers")

    st.markdown("")

paper_id = st.session_state.query_option.split("] ")[0]
paper_id = paper_id[1:]
full_text = ""
for abs in papers[paper_id].abstract:
    full_text += abs
    full_text += " "
full_text = st.session_state.query_option + "\n" + full_text
sentences = int(len(full_text)/250 + 2)
st.text_area("Full query text", full_text, height=30*sentences)

if(st.session_state.loss_func == "Cross-Entropy-Loss"):
    candidates = QBERetrieveSciArticles(
        st.session_state.model, st.session_state.facet, paper_id, loss_fn = "NLLLoss", top=True, ret_k=st.session_state.top_N)
else:
    candidates = QBERetrieveSciArticles(
        st.session_state.model, st.session_state.facet, paper_id, loss_fn = "KLDivLoss", top=True, ret_k=st.session_state.top_N)

st.subheader("Query Results")

for i in range(min(st.session_state.top_N, len(candidates))):
    pid = candidates[i][0]
    title = "Rank " + str(i + 1) + ": [" + pid + "] " + papers[pid].title
    full_text = ""
    for abs in papers[pid].abstract:
        full_text += abs
        full_text += " "
    if str(papers[pid].metadata["doi"]) != "None":
        full_text += "\n\nLink to the paper: www.doi.org/" + str(papers[pid].metadata["doi"])
    with st.expander(title, expanded=False):	
        st.write(full_text) 
