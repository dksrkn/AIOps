import os
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def build_report_input(
    eval_result: Dict[str, Any],
    preview_rows: List[Dict[str, Any]],
    error_by_building_type: List[Dict[str, Any]],
    rmse_before: float | None = None,
    rmse_after: float | None = None,
    retrain_executed: bool = False,
    model_replaced: bool = False,
) -> Dict[str, Any]:
    return {
        "rmse_before": rmse_before,
        "rmse_after": rmse_after if rmse_after is not None else eval_result.get("rmse"),
        "rmse_threshold": eval_result.get("rmse_threshold"),
        "retrain_required": eval_result.get("retrain_required"),
        "retrain_executed": retrain_executed,
        "model_replaced": model_replaced,
        "preview_rows": preview_rows[:10],
        "error_by_building_type": error_by_building_type,
    }


def generate_report(
    eval_result: Dict[str, Any],
    preview_rows: List[Dict[str, Any]],
    error_by_building_type: List[Dict[str, Any]],
    rmse_before: float | None = None,
    rmse_after: float | None = None,
    retrain_executed: bool = False,
    model_replaced: bool = False,
) -> Dict[str, Any]:
    payload = build_report_input(
        eval_result=eval_result,
        preview_rows=preview_rows,
        error_by_building_type=error_by_building_type,
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        retrain_executed=retrain_executed,
        model_replaced=model_replaced,
    )

    prompt = f"""
    너는 전력 소비량 기반 시계열 예측 모델을 평가하는 데이터 사이언티스트다.

    아래 입력 데이터를 바탕으로 사용자에게 보여줄 최종 분석 보고서를 작성하라.

    [입력 데이터]
    {json.dumps(payload, ensure_ascii=False, indent=2)}

    [분석 지시]
    1. 전체 분석 결과를 2~3문장으로 요약하라.
    2. 재학습 전후 RMSE 차이를 중심으로 성능 분석 변화를 설명하라.
    3. 현재 모델 성능이 기준 대비 어떤 상태인지 설명하라.
    4. error_by_building_type를 바탕으로
    - 오차가 가장 큰 건물 유형
    - 상대적으로 안정적인 건물 유형
    을 반드시 언급하라.
    5. retrain_executed, model_replaced 값을 바탕으로 재학습 상태를 설명하라.
    6. 향후 모델 개선 또는 운영 측면에서 필요한 권장 조치를 작성하라.
    7. 사용자가 이해하기 쉬운 한국어로 작성하라.
    8. 마크다운, 코드블록 없이 JSON만 반환하라.
    9. 모든 값은 문자열로 작성하라.

    [제외 항목]
    - preview_rows를 바탕으로 예측 패턴 분석을 별도 항목으로 작성하지 마라.
    - 주요 강점 항목을 작성하지 마라.
    - 주요 한계 항목을 작성하지 마라.

    [반환 JSON 형식]
    {{
    "title": "전력 소비 예측 성능 분석 보고서",
    "summary": "...",
    "performance_analysis": "...",
    "building_type_analysis": "...",
    "action": "..."
    }}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    text = response.output_text
    return json.loads(text)