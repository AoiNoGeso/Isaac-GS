"""設定クラス -> 観測項目クラス のレジストリ"""

from __future__ import annotations

from typing import Any

from envs.observations.base import ObsTerm, ObsTermCfg

OBS_TERMS: dict[type[ObsTermCfg], type] = {}


def register_obs_term(cfg_cls: type[ObsTermCfg]):
    """観測項目クラスをその設定クラスに紐づけて登録するデコレータ"""

    def _decorator(term_cls: type) -> type:
        OBS_TERMS[cfg_cls] = term_cls
        return term_cls

    return _decorator


def build_term(cfg: ObsTermCfg, env: Any) -> ObsTerm:
    """設定クラスから対応する観測項目を生成する。未登録ならKeyErrorを分かりやすく送出"""
    cfg_cls = type(cfg)
    term_cls = OBS_TERMS.get(cfg_cls)
    if term_cls is None:
        raise KeyError(
            f"観測項目の設定クラス {cfg_cls.__name__} に対応するtermが登録されていません。"
            f"登録済み: {[c.__name__ for c in OBS_TERMS]}"
        )
    return term_cls(cfg, env)
