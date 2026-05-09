import math
import typing as T
import unittest

from pytorch3d.ops import knn_points
import torch

from plibs.ppoint import (
    PackedPoint,
    cross_softmax_attention_with_packed_kv,
    cross_softmax_attention_with_packed_qkv,
    localized_knn_cross_softmax_attention_with_packed_qkv,
    localized_knn_self_softmax_attention_packed,
    voxel_downsampling,
    voxel_windowed_cross_softmax_attention_with_packed_qkv,
    voxel_windowed_self_softmax_attention,
)


class TestPPoint(unittest.TestCase):
    def create_points(self, b: int, grid_size: int):
        """Create a dense grid of points"""
        xs = torch.arange(grid_size).float() + 0.5  # +0.5 to be at cell center
        Y, X = torch.meshgrid(xs, xs, indexing="ij")  # (grid_size, grid_size)
        xy = torch.stack([X, Y], dim=-1).reshape(-1, 2)  # (n, 2)

        all_xys = []
        for i in range(b):
            all_xys.append(xy + 100 * i)
        all_xys = torch.stack(all_xys, dim=0)  # (b, n, 2)
        return all_xys

    def get_bijk_feature(
        self,
        packed_coord: PackedPoint,
        cell_width: float,
        d: int,
    ):
        info_dict = packed_coord.get_bijk_info(
            cell_width=cell_width,
            shift=0,
            save_to_cache=False,
        )
        packed_feature = torch.zeros(
            packed_coord.bn, d, dtype=packed_coord.dtype, device=packed_coord.device
        )  # (bn, d)
        for didx in range(d):
            packed_feature[:, didx] = (info_dict["linear_idx"]).to(dtype=packed_coord.dtype)
        return packed_feature  # (bn, d)

    def get_b_feature(
        self,
        packed_coord: PackedPoint,
        d: int,
    ):
        fs = []
        for i in range(packed_coord.batch_size):
            f = (
                torch.ones(packed_coord.seq_lens[i], d, dtype=packed_coord.dtype, device=packed_coord.device) * i
            )  # (ni, d)
            fs.append(f)
        packed_feature = torch.cat(fs, dim=0)  # (bn, d)
        return packed_feature  # (bn, d)

    def test_packed_point(self):
        xys = self.create_points(b=3, grid_size=4)  # (b, n, 2)
        b, n, _dn = xys.shape
        pp = PackedPoint(
            coord=xys.reshape(-1, 2),  # (bn, 2)
            seq_lens=torch.tensor([n] * b).long(),
        )
        assert pp.batch_size == b, f"{pp.batch_size=}, {b=}"
        assert tuple(pp.seq_lens) == tuple([n] * b), f"{tuple(pp.seq_lens)=}, {tuple([n] * b)=}"
        assert pp.bn == b * n, f"{pp.bn=}, {(b * n)=}"
        assert torch.allclose(pp.batch_idxs, torch.arange(b).unsqueeze(-1).expand(-1, n).reshape(-1))

    def test_voxel_downsampling(self):
        xys = self.create_points(b=3, grid_size=4)  # (b, n, 2)
        b, n, _dn = xys.shape
        pp = PackedPoint(
            coord=xys.reshape(-1, 2),  # (bn, 2)
            seq_lens=torch.tensor([n] * b).long(),
        )
        pf = self.get_bijk_feature(
            packed_coord=pp,
            cell_width=2,
            d=1,
        )  # (pp.bn, d)
        out_dict = voxel_downsampling(
            packed_coord=pp,
            packed_feature=pf,
            cell_width=2,
            shift=0,
            save_to_cache=True,
        )
        new_pp = out_dict["packed_coord"]
        new_feature = out_dict["packed_feature"]

        new_info_dict = new_pp.get_bijk_info(
            cell_width=2,
            shift=0,
            save_to_cache=False,
        )
        assert torch.allclose(new_feature, new_info_dict["linear_idx"][:, None].to(dtype=new_feature.dtype))

    def test_voxel_self_attention(self):
        if not torch.cuda.is_available():
            print(f"not testing due to CUDA availability")
            return

        device = torch.device("cuda")
        xys = self.create_points(b=3, grid_size=16)  # (b, n, 2)
        b, n, _dn = xys.shape
        pp = PackedPoint(
            coord=xys.reshape(-1, 2),  # (bn, 2)
            seq_lens=torch.tensor([n] * b).long(),
        )
        pp.to(device=device)

        dim_qk = 8
        dim_v = 8
        cell_width = 4
        packed_query = torch.randn(pp.bn, 1, dim_qk, device=device)  # (bn, 1, dim_qk)
        packed_key = torch.randn(pp.bn, 1, dim_qk, device=device)  # (bn, 1, dim_qk)
        packed_value = self.get_bijk_feature(
            packed_coord=pp,
            cell_width=cell_width,
            d=dim_v,
        )  # (bn, dim_v)
        packed_value = packed_value.unsqueeze(-2)  # (bn, 1, dim_v)

        packed_out = voxel_windowed_self_softmax_attention(
            packed_coord=pp,
            packed_query=packed_query,
            packed_key=packed_key,
            packed_value=packed_value,
            cell_width=cell_width,
            shift=0,
            save_to_cache=True,
        )

        assert torch.allclose(packed_out.squeeze(-2), packed_value.to(dtype=packed_out.dtype).squeeze(-2))

    def test_cross_attention(self):
        if not torch.cuda.is_available():
            print(f"not testing due to CUDA availability")
            return

        device = torch.device("cuda")
        xys = self.create_points(b=3, grid_size=16)  # (b, n, 2)
        b, n, _dn = xys.shape
        pp = PackedPoint(
            coord=xys.reshape(-1, 2),  # (bn, 2)
            seq_lens=torch.tensor([n] * b).long(),
        )
        pp.to(device=device)

        m = 10  # number of query
        dim_qk = 8
        dim_v = 8
        cell_width = 4
        query = torch.randn(b, m, 1, dim_qk, device=device)  # (b, m, 1, dim_qk)
        packed_key = torch.randn(pp.bn, 1, dim_qk, device=device)  # (bn, 1, dim_qk)
        packed_value = self.get_b_feature(
            packed_coord=pp,
            d=dim_v,
        )  # (bn, dim_v)
        packed_value = packed_value.unsqueeze(-2)  # (bn, 1, dim_v)

        out = cross_softmax_attention_with_packed_kv(
            query=query,
            packed_kv_coord=pp,
            packed_key=packed_key,
            packed_value=packed_value,
            save_to_cache=True,
        )  # (b, m, h=1, d), m for number of queries, h for head

        gt = torch.arange(b, dtype=out.dtype, device=out.device).reshape(b, 1, 1, 1).expand_as(out)
        assert torch.allclose(out, gt)

    def test_cross_attention_with_packed_qkv(self):
        if not torch.cuda.is_available():
            print(f"not testing due to CUDA availability")
            return

        device = torch.device("cuda")
        # kv
        xys = self.create_points(b=3, grid_size=16)  # (b, n, 2)
        b, n, _dn = xys.shape
        pp = PackedPoint(
            coord=xys.reshape(-1, 2),  # (bn, 2)
            seq_lens=torch.tensor([n] * b).long(),
        )
        pp.to(device=device)

        # q
        ms = [3, 5, 2]
        assert len(ms) == b
        bm = sum(ms)
        qq = PackedPoint(
            coord=torch.randn(bm, 2),  # (bn, 2)
            seq_lens=torch.tensor(ms).long(),
        )
        qq.to(device=device)

        dim_qk = 8
        dim_v = 8
        packed_query = torch.randn(qq.bn, 1, dim_qk, device=device)  # (bm, 1, dim_qk)
        packed_key = torch.randn(pp.bn, 1, dim_qk, device=device)  # (bn, 1, dim_qk)
        packed_value = self.get_b_feature(
            packed_coord=pp,
            d=dim_v,
        )  # (bn, dim_v)
        packed_value = packed_value.unsqueeze(-2)  # (bn, 1, dim_v)

        out = cross_softmax_attention_with_packed_qkv(
            packed_query_coord=qq,
            packed_query=packed_query,
            packed_kv_coord=pp,
            packed_key=packed_key,
            packed_value=packed_value,
            save_to_cache=True,
        )  # (bm, h=1, d)

        gt = []
        for ib in range(b):
            gt += [ib] * ms[ib]
        gt = torch.tensor(gt, dtype=out.dtype, device=out.device)  # (bm,)
        gt = gt.reshape(bm, 1, 1).expand_as(out)
        assert torch.allclose(out, gt)

    def test_localized_knn_cross_softmax_attention_with_packed_qkv(self):
        if not torch.cuda.is_available():
            print(f"not testing due to CUDA availability")
            return

        device = torch.device("cuda")

        # kv
        b = 3
        ns = [16, 7, 32]
        ms = [4, 2, 3]
        assert len(ms) == b, f"{len(ms)=}, {b=}"
        assert len(ns) == b, f"{len(ns)=}, {b=}"
        bn = sum(ns)
        bm = sum(ms)
        dim_qk = 8

        packed_keys = torch.randn(bn, 1, dim_qk, device=device)  # not matter
        packed_queries = torch.randn(bm, 1, dim_qk, device=device)  # not matter

        packed_coord_q = []
        packed_coord_kv = []
        packed_value = []

        current_kidx = 0
        for ib in range(b):
            coord_k = torch.randn(ns[ib], 2, device=device)  # (ni, 2)
            # construct q from k (so we make sure query attend to at least a key (itself))
            ridxs = torch.randperm(ns[ib], device=device)[: ms[ib]]
            coord_q = coord_k[ridxs]  # (mi, 2)

            # knn
            knn_out = knn_points(
                p1=coord_k.unsqueeze(0),  # (1, ni, 2)
                p2=coord_q.unsqueeze(0),  # (1, mi, 2)
                K=1,
            )
            value = knn_out.idx.float() + current_kidx  # (1, ni)
            value = value.reshape(ns[ib])

            packed_coord_q.append(coord_q)
            packed_coord_kv.append(coord_k)
            packed_value.append(value)
            current_kidx += ms[ib]

        packed_coord_q = torch.cat(packed_coord_q, dim=0)  # (bm, 2)
        packed_coord_kv = torch.cat(packed_coord_kv, dim=0)  # (bn, 2)
        packed_value = torch.cat(packed_value, dim=0)  # (bn,)
        packed_value = packed_value.reshape(bn, 1, 1).expand(bn, 1, dim_qk)

        packed_coord_q = PackedPoint(
            coord=packed_coord_q,
            seq_lens=torch.tensor(ms, dtype=torch.long, device=device),
        )
        packed_coord_kv = PackedPoint(
            coord=packed_coord_kv,
            seq_lens=torch.tensor(ns, dtype=torch.long, device=device),
        )

        out = localized_knn_cross_softmax_attention_with_packed_qkv(
            packed_query_coord=packed_coord_q,
            packed_query=packed_queries,
            packed_kv_coord=packed_coord_kv,
            packed_key=packed_keys,
            packed_value=packed_value,
        )  # (bm, 1, dim_qk)

        gt = torch.arange(bm, dtype=out.dtype, device=out.device).reshape(bm, 1, 1).expand_as(out)
        assert torch.allclose(out, gt)

    def test_voxel_windowed_cross_softmax_attention_with_packed_qkv(self):
        if not torch.cuda.is_available():
            print(f"not testing due to CUDA availability")
            return

        device = torch.device("cuda")

        # kv
        b = 3
        ns = [16, 7, 32]
        ms = [20, 3, 19]
        cell_width = 0.2
        shift = 0
        assert len(ms) == b, f"{len(ms)=}, {b=}"
        assert len(ns) == b, f"{len(ns)=}, {b=}"
        bn = sum(ns)
        bm = sum(ms)
        dim_qk = 8

        packed_keys = torch.randn(bn, 1, dim_qk, device=device)  # not matter
        packed_queries = torch.randn(bm, 1, dim_qk, device=device)  # not matter

        packed_coord_q = []
        packed_coord_kv = []
        packed_value = []
        linear_idx_qs = []

        grid_size = math.ceil(2 / cell_width)

        current_kidx = 0
        for ib in range(b):
            coord_k = torch.rand(ns[ib], 2, device=device) * 2 - 1  # (ni, 2) [-1, 1]
            coord_q = torch.rand(ms[ib], 2, device=device) * 2 - 1  # (mi, 2) [-1, 1]

            ij_k = torch.floor((coord_k + shift - (-1) / cell_width))  # (ni, 2ij)
            ij_q = torch.floor((coord_q + shift - (-1) / cell_width))  # (mi, 2ij)
            linear_idx_k = ib * (grid_size**2) + ij_k[..., 1] * grid_size + ij_k[..., 0]  # (ni,)
            linear_idx_q = ib * (grid_size**2) + ij_q[..., 1] * grid_size + ij_q[..., 0]  # (mi,)

            # This makes sure that points in the same cell have same values, namely cell indices.
            # Since query will only attend to the points in the same cell as the query,
            # this makes the GT output of cross-attention to be the cell's index.
            value = linear_idx_k

            packed_coord_q.append(coord_q)
            packed_coord_kv.append(coord_k)
            packed_value.append(value)
            linear_idx_qs.append(linear_idx_q)
            current_kidx += ms[ib]

        packed_coord_q = torch.cat(packed_coord_q, dim=0)  # (bm, 2)
        packed_coord_kv = torch.cat(packed_coord_kv, dim=0)  # (bn, 2)
        packed_value = torch.cat(packed_value, dim=0)  # (bn,)
        packed_value = packed_value.reshape(bn, 1, 1).expand(bn, 1, dim_qk)
        linear_idx_qs = torch.cat(linear_idx_qs, dim=0)  # (bm,)

        packed_coord_q = PackedPoint(
            coord=packed_coord_q,
            seq_lens=torch.tensor(ms, dtype=torch.long, device=device),
        )
        packed_coord_kv = PackedPoint(
            coord=packed_coord_kv,
            seq_lens=torch.tensor(ns, dtype=torch.long, device=device),
        )

        out = voxel_windowed_cross_softmax_attention_with_packed_qkv(
            packed_query_coord=packed_coord_q,
            packed_query=packed_queries,
            packed_kv_coord=packed_coord_kv,
            packed_key=packed_keys,
            packed_value=packed_value,
            cell_width=cell_width,
            shift=shift,
        )  # (bm, 1, dim_qk)

        gt = linear_idx_qs.to(dtype=out.dtype).reshape(bm, 1, 1).expand_as(out)
        assert torch.logical_or(
            torch.isclose(out, gt),
            torch.isclose(out, torch.zeros_like(out)),  # if no key in the same cell as query
        ).all()

    def test_localized_knn_self_softmax_attention_packed(self):
        if not torch.cuda.is_available():
            print(f"not testing due to CUDA availability")
            return

        device = torch.device("cuda")

        # kv
        b = 3
        ns = [16, 7, 32]
        assert len(ns) == b, f"{len(ns)=}, {b=}"
        bn = sum(ns)
        dim_qk = 8
        cache_name = "debug"
        k = 5

        packed_coord_kv = torch.randn(bn, 2, device=device)  # (bn, 2)
        packed_queries = torch.randn(bn, 1, dim_qk, device=device)  # not matter
        packed_keys = torch.randn(bn, 1, dim_qk, device=device)  # not matter

        packed_coord_kv = PackedPoint(
            coord=packed_coord_kv,
            seq_lens=torch.tensor(ns, dtype=torch.long, device=device),
        )

        info_dict = packed_coord_kv.get_localized_self_knn_info(
            k=k,
            cache_name=cache_name,
            use_cached=True,
            save_to_cache=True,
            attn_backend="xformers",
            debug=True,
        )
        kidxs = info_dict["kidxs"]  # (bn,)
        packed_value = (
            kidxs.to(dtype=packed_keys.dtype, device=packed_keys.device).reshape(bn, 1, 1).expand(bn, 1, dim_qk)
        )

        out = localized_knn_self_softmax_attention_packed(
            packed_coord=packed_coord_kv,
            packed_query=packed_queries,
            packed_key=packed_keys,
            packed_value=packed_value,
            k=k,
            use_cached=True,
            cache_name=cache_name,
        )  # (bm, 1, dim_qk)

        gt = kidxs.reshape(bn, 1, 1).expand_as(out).to(dtype=out.dtype, device=out.device)

        # assert torch.logical_or(
        #     torch.isclose(out, gt),
        #     torch.isclose(out, torch.zeros_like(out)),  # for checking the situation that input has shape of (b, 0, d)
        # ).all()
        assert torch.allclose(out, gt)


if __name__ == "__main__":
    unittest.main()
