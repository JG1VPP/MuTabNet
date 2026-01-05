from time import perf_counter_ns
from typing import List

import torch
import torch.nn as nn

from mutab.models.factory import MODELS, build


@MODELS.register_module()
class TableDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        html_decoder,
        cell_decoder,
        html_fetcher,
        bbox_locator,
        num_emb_html: int,
        num_emb_cell: int,
        max_len_html: int,
        max_len_cell: int,
        SOC_HTML: List[int],
        SOS_HTML: int,
        EOS_HTML: int,
        SOS_CELL: int,
        EOS_CELL: int,
        SEP_CELL: int,
        **kwargs,
    ):
        super().__init__()

        # parameters
        html_decoder.update(d_model=d_model)
        cell_decoder.update(d_model=d_model)
        html_fetcher.update(d_model=d_model)
        bbox_locator.update(d_model=d_model)

        # alphabet
        html_decoder.update(num_emb=num_emb_html)
        cell_decoder.update(num_emb=num_emb_cell)

        # capacity
        html_decoder.update(max_len=max_len_html)
        cell_decoder.update(max_len=max_len_cell)

        # special tokens
        html_decoder.update(SOS=SOS_HTML)
        html_decoder.update(EOS=EOS_HTML)
        html_decoder.update(SEP=EOS_HTML)

        cell_decoder.update(SOS=SOS_CELL)
        cell_decoder.update(EOS=EOS_CELL)
        cell_decoder.update(SEP=SEP_CELL)

        html_fetcher.update(SOC=SOC_HTML)
        html_fetcher.update(EOS=EOS_HTML)

        # input channels
        html_decoder.update(d_input=d_model + 2)
        cell_decoder.update(d_input=d_model * 2)

        # networks
        self.html = build(html_decoder, **kwargs)
        self.cell = build(cell_decoder, **kwargs)
        self.grid = build(html_fetcher, **kwargs)
        self.bbox = build(bbox_locator, **kwargs)

        # LtoR or RtoL
        self.register_buffer("LtoR", torch.eye(2)[0])
        self.register_buffer("RtoL", torch.eye(2)[1])

    def forward(self, img, html, back, cell, **kwargs):
        # ground truth
        html = html.to(img.device)
        back = back.to(img.device)
        cell = cell.to(img.device)

        # remove [EOS]
        s_html = html[:, :-1]
        e_back = back[:, :-1]
        s_cell = cell[:, :-1]

        # remove [SOS]
        e_html = html[:, 1::]

        # LtoR or RtoL
        h_LtoR = self.LtoR.expand(len(img), 1, 2)
        h_RtoL = self.RtoL.expand(len(img), 1, 2)

        # structure prediction
        h_html, o_html = self.html(img, s_html, h_LtoR)
        h_back, o_back = self.html(img, e_back, h_RtoL)

        # structure refinement
        h_html, h_grid = self.grid(img, h_html, e_html)
        h_grid, o_bbox = self.bbox(img, h_html, h_grid)

        # character prediction
        h_cell, o_cell = self.cell(img, s_cell, h_grid)

        return dict(
            html=o_html,
            back=o_back,
            cell=o_cell,
            bbox=o_bbox,
        )

    def predict(self, img, time: int):
        # LtoR
        h_LtoR = self.LtoR.expand(len(img), 1, 2)

        # structure prediction
        h_html, o_html = self.html.predict(img, h_LtoR)
        t_html = perf_counter_ns()

        # structure refinement
        h_html, h_grid = self.grid(img, h_html, o_html)
        h_grid, o_bbox = self.bbox(img, h_html, h_grid)
        t_bbox = perf_counter_ns()

        # character prediction
        h_cell, o_cell = self.cell.predict(img, h_grid)
        t_cell = perf_counter_ns()

        time = dict(init=time, html=t_html, bbox=t_bbox, cell=t_cell)
        return dict(html=o_html, cell=o_cell, bbox=o_bbox, time=time)
