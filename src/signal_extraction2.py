"""Alternative signal extraction module that preserves cleaned document sections."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.signal_10k_def_func import (
    gen_feat_ch_full_len,
    gen_feat_ch_item_1a_len,
    gen_feat_ch_item_1b_len,
    gen_feat_ch_item_3_len,
    gen_feat_full_cos_1gram,
    gen_feat_full_cos_2gram,
    gen_feat_full_jac_1gram,
    gen_feat_full_jac_2gram,
    gen_feat_item_1a_lev,
    gen_feat_item_7_lev,
    gen_feat_lm_postive,
    gen_feat_lm_uncertainty,
    gen_feat_lm_litigious,
    gen_feat_word2vec,
)
from src.signal_10q_def_func import (
    gen_feat_ch_full_len_10q,
    gen_feat_full_cos_1gram_10q,
    gen_feat_full_jac_1gram_10q,
    gen_feat_lm_postive_10q,
    gen_feat_word2vec_10q,
)
from src.signal_extraction import SignalExtraction
from src.text_processing import clean_doc1, clean_doc2, find_item_pos, get_item_ptrn1
from src.util import log, save_pkl
import src.constants as const


class SignalExtractionWithDocs(SignalExtraction):
    """Extends :class:`SignalExtraction` to retain cleaned document sections."""

    def __init__(self):
        super().__init__()
        self.docs_10k: Optional[pd.DataFrame] = None
        self.docs_10q: Optional[pd.DataFrame] = None

    def _request_txt(self, url: str) -> str:
        session = requests.Session()
        retry = Retry(
            connect=self.config["retry_connect"],
            backoff_factor=self.config["retry_backoff_factor"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        headers = {
            "Host": "www.sec.gov",
            "Connection": "close",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": f"{self.config['edgar_user_agent']}{int(float(np.random.rand(1)) * 1e7)}",
        }
        return session.get(url, headers=headers).text

    def _flatten_docs(
        self,
        cik: str,
        df: pd.DataFrame,
        docs: Dict[str, Dict[str, str]],
        form_type: str,
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for doc_id, section_dict in docs.items():
            meta = df.loc[df.doc_id == doc_id]
            if meta.empty:
                continue
            filing_date = meta.iloc[0]["filing_date"]
            entity = meta.iloc[0]["entity"]
            for section_name, text in section_dict.items():
                rows.append(
                    {
                        "cik": str(cik),
                        "doc_id": doc_id,
                        "filing_date": filing_date,
                        "entity": entity,
                        "section": section_name,
                        "text": text,
                        "form_type": form_type,
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "cik",
                    "doc_id",
                    "filing_date",
                    "entity",
                    "section",
                    "text",
                    "form_type",
                ]
            )
        return pd.DataFrame(rows)

    def gen_signal_10k_with_docs(self, cik: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        log(f"{cik}: Extracting 10-K signal with document retention...")
        df = (
            self.master_idx_10k.loc[lambda x: x.cik == cik]
            .sort_values("filing_date")
            .reset_index(drop=True)
        )
        docs: Dict[str, Dict[str, str]] = {}
        for i in range(len(df)):
            url = df.iloc[i]["url_10k"]
            doc_id = df.iloc[i]["doc_id"]
            txt = self._request_txt(url)
            txt = BeautifulSoup(txt, "lxml").get_text("|", strip=True)
            txt = clean_doc1(txt)
            item_pos = find_item_pos(txt)
            doc_dict = {"full": txt[item_pos.iloc[0]["item_1_pos_start"] :]}  # type: ignore[index]
            item_ptrn1 = get_item_ptrn1()
            for item in item_ptrn1:
                doc_dict[item] = txt[
                    item_pos.iloc[0][f"{item}_pos_start"] : item_pos.iloc[0][f"{item}_pos_end"]
                ]
            for key in doc_dict:
                doc_dict[key] = clean_doc2(doc_dict[key])
            docs[doc_id] = doc_dict

        sections_df = self._flatten_docs(cik, df, docs, "10-K")

        feat_vecs = [pd.Series(list(docs.keys())).rename("doc_id")]
        feat_vecs += [
            gen_feat_ch_full_len(docs),
            gen_feat_ch_item_1a_len(docs),
            gen_feat_ch_item_1b_len(docs),
            gen_feat_ch_item_3_len(docs),
            gen_feat_full_cos_1gram(docs, self.global_tfidf_1g),
            gen_feat_full_cos_2gram(docs, self.global_tfidf_2g),
            gen_feat_full_jac_1gram(docs),
            gen_feat_full_jac_2gram(docs),
            gen_feat_item_1a_lev(docs),
            gen_feat_item_7_lev(docs),
            gen_feat_lm_postive(docs, self.positive_word_list, self.negative_word_list),
            gen_feat_lm_uncertainty(docs, self.uncertainty_word_list),
            gen_feat_lm_litigious(docs, self.litigious_word_list),
            gen_feat_word2vec(
                docs,
                self.global_tfidf_1g,
                self.tfidf_1g_wv_idx,
                self.wv_subset,
            ),
        ]
        if self.config["gpu_enabled"]:
            from src.signal_10k_def_func import (
                gen_feat_sen_enc,
                gen_feat_item_sentiment,
                gen_feat_fls_sentiment,
            )

            feat_vecs += [
                gen_feat_sen_enc(docs, self.st_model),
                gen_feat_item_sentiment(docs, self.fb_tokenizer, self.fb_model),
                gen_feat_fls_sentiment(docs, self.fb_tokenizer, self.fb_model),
            ]

        feats = pd.concat(feat_vecs, axis=1)
        return feats, sections_df

    def gen_signal_10q_with_docs(self, cik: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        log(f"{cik}: Extracting 10-Q signal with document retention...")
        df = (
            self.master_idx_10q.loc[lambda x: x.cik == cik]
            .sort_values("filing_date")
            .reset_index(drop=True)
        )
        docs: Dict[str, Dict[str, str]] = {}
        for i in range(len(df)):
            url = df.iloc[i]["url_10q"]
            doc_id = df.iloc[i]["doc_id"]
            txt = self._request_txt(url)
            txt = BeautifulSoup(txt, "lxml").get_text("|", strip=True)
            txt = clean_doc1(txt)
            doc_dict = {"full": clean_doc2(txt)}
            docs[doc_id] = doc_dict

        sections_df = self._flatten_docs(cik, df, docs, "10-Q")

        doc_pairs = self.get_10q_doc_pairs(docs)
        feat_vecs = [doc_pairs.doc_id]
        feat_vecs += [
            gen_feat_ch_full_len_10q(docs, doc_pairs),
            gen_feat_full_cos_1gram_10q(docs, doc_pairs),
            gen_feat_full_jac_1gram_10q(docs, doc_pairs),
            gen_feat_word2vec_10q(
                docs,
                doc_pairs,
                self.global_tfidf_1g,
                self.tfidf_1g_wv_idx,
                self.wv_subset,
            ),
            gen_feat_lm_postive_10q(
                docs,
                doc_pairs,
                self.positive_word_list,
                self.negative_word_list,
            ),
        ]
        feats = pd.concat(feat_vecs, axis=1)
        return feats, sections_df

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def gen_signal_10k(self, cik: str):  # type: ignore[override]
        feats, sections = self.gen_signal_10k_with_docs(cik)
        if sections is not None and not sections.empty:
            if self.docs_10k is None:
                self.docs_10k = sections
            else:
                self.docs_10k = pd.concat([self.docs_10k, sections], ignore_index=True)
        return feats

    def gen_signal_10q(self, cik: str):  # type: ignore[override]
        feats, sections = self.gen_signal_10q_with_docs(cik)
        if sections is not None and not sections.empty:
            if self.docs_10q is None:
                self.docs_10q = sections
            else:
                self.docs_10q = pd.concat([self.docs_10q, sections], ignore_index=True)
        return feats

    def gen_signal_10k_all_stocks(self):  # type: ignore[override]
        self.docs_10k = None
        results = [self.gen_signal_10k_with_docs(cik) for cik in self.master_idx_10k.cik.unique()]
        feat_frames = [res[0] for res in results if isinstance(res[0], pd.DataFrame)]
        doc_frames = [res[1] for res in results if not res[1].empty]

        if feat_frames:
            feats = pd.concat(feat_frames).sort_values("doc_id").reset_index(drop=True)
            df = self.master_idx_10k[["doc_id", "cik", "entity", "filing_date"]].drop_duplicates()
            feats = feats.merge(df, how="inner", on="doc_id")
            feats = feats.merge(self.cik_map, how="inner", on="cik")
            cols = [c for c in feats if c[:5] == "feat_"]
            feats = feats[[c for c in feats if c not in cols] + cols]
            self.feats_10k = feats
            save_pkl(feats, os.path.join(const.INTERIM_DATA_PATH, "feats_10k.pkl"))

        if doc_frames:
            docs = pd.concat(doc_frames).reset_index(drop=True)
            self.docs_10k = docs if self.docs_10k is None else pd.concat([self.docs_10k, docs], ignore_index=True)
            save_pkl(self.docs_10k, os.path.join(const.INTERIM_DATA_PATH, "docs_10k_sections.pkl"))
            log(f"Saved 10-K document sections with shape {self.docs_10k.shape}")

    def gen_signal_10q_all_stocks(self):  # type: ignore[override]
        self.docs_10q = None
        results = [self.gen_signal_10q_with_docs(cik) for cik in self.master_idx_10q.cik.unique()]
        feat_frames = [res[0] for res in results if isinstance(res[0], pd.DataFrame)]
        doc_frames = [res[1] for res in results if not res[1].empty]

        if feat_frames:
            feats = pd.concat(feat_frames).sort_values("doc_id").reset_index(drop=True)
            df = self.master_idx_10q[["doc_id", "cik", "entity", "filing_date"]].drop_duplicates()
            feats = feats.merge(df, how="inner", on="doc_id")
            feats = feats.merge(self.cik_map, how="inner", on="cik")
            cols = [c for c in feats if c[:5] == "feat_"]
            feats = feats[[c for c in feats if c not in cols] + cols]
            self.feats_10q = feats
            save_pkl(feats, os.path.join(const.INTERIM_DATA_PATH, "feats_10q.pkl"))

        if doc_frames:
            docs = pd.concat(doc_frames).reset_index(drop=True)
            self.docs_10q = docs if self.docs_10q is None else pd.concat([self.docs_10q, docs], ignore_index=True)
            save_pkl(self.docs_10q, os.path.join(const.INTERIM_DATA_PATH, "docs_10q_sections.pkl"))
            log(f"Saved 10-Q document sections with shape {self.docs_10q.shape}")
