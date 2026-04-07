const uploadBtn = document.getElementById("uploadBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const csvFileInput = document.getElementById("csvFile");
const uploadPanel = document.getElementById("uploadPanel");
const uploadToolbar = document.querySelector(".upload-toolbar");
const uploadStatus = document.getElementById("uploadStatus");

const rmseValue = document.getElementById("rmseValue");
const retrainStatus = document.getElementById("retrainStatus");
const summaryMessage = document.getElementById("summaryMessage");
const retrainMessage = document.getElementById("retrainMessage");

const errorTableBody = document.getElementById("errorTableBody");

const approvalSection = document.getElementById("approvalSection");
const approvalMessage = document.getElementById("approvalMessage");
const approveBtn = document.getElementById("approveBtn");
const rejectBtn = document.getElementById("rejectBtn");

const kpiEnergy = document.getElementById("kpiEnergy");
const kpiPeak = document.getElementById("kpiPeak");
const kpiSaving = document.getElementById("kpiSaving");
const kpiConfidence = document.getElementById("kpiConfidence");
const kpiBuildingType = document.getElementById("kpiBuildingType");
const kpiVariables = document.getElementById("kpiVariables");

let chartBeforeInstance = null;
let chartAfterInstance = null;

let uploadPanelVisible = false;
let pendingMockResult = null;

const mockData = {
  rmse: 359.85,
  rmse_before: 512.4,
  rmse_after: 359.85,

  retrain_required: true,
  retrain_executed: true,
  model_replaced: false,

  message:
    "RMSE 기준 초과로 재학습이 수행되었습니다. 결과를 확인한 후 운영 모델 반영 여부를 결정할 수 있습니다.",

  preview_before: [
    { date_time: "2024-08-14 13:00", actual: 6320.4, predicted: 6050.2 },
    { date_time: "2024-08-14 14:00", actual: 6402.1, predicted: 6124.6 },
    { date_time: "2024-08-14 15:00", actual: 6550.3, predicted: 6201.3 },
    { date_time: "2024-08-14 16:00", actual: 6481.2, predicted: 6188.4 },
    { date_time: "2024-08-14 17:00", actual: 5902.7, predicted: 6030.1 },
    { date_time: "2024-08-14 18:00", actual: 6612.9, predicted: 6115.7 },
    { date_time: "2024-08-14 19:00", actual: 6458.3, predicted: 5960.8 },
    { date_time: "2024-08-14 20:00", actual: 5200.1, predicted: 5854.9 },
    { date_time: "2024-08-14 21:00", actual: 3710.8, predicted: 5682.4 },
    { date_time: "2024-08-14 22:00", actual: 5015.4, predicted: 5488.7 },
    { date_time: "2024-08-14 23:00", actual: 5578.0, predicted: 5305.2 },
    { date_time: "2024-08-15 00:00", actual: 5007.1, predicted: 5154.3 }
  ],

  preview_after: [
    { date_time: "2024-08-14 13:00", actual: 6320.4, predicted: 6211.8 },
    { date_time: "2024-08-14 14:00", actual: 6402.1, predicted: 6338.9 },
    { date_time: "2024-08-14 15:00", actual: 6550.3, predicted: 6477.2 },
    { date_time: "2024-08-14 16:00", actual: 6481.2, predicted: 6418.5 },
    { date_time: "2024-08-14 17:00", actual: 5902.7, predicted: 6081.4 },
    { date_time: "2024-08-14 18:00", actual: 6612.9, predicted: 6140.1 },
    { date_time: "2024-08-14 19:00", actual: 6458.3, predicted: 5890.4 },
    { date_time: "2024-08-14 20:00", actual: 5200.1, predicted: 5703.8 },
    { date_time: "2024-08-14 21:00", actual: 3710.8, predicted: 5521.6 },
    { date_time: "2024-08-14 22:00", actual: 5015.4, predicted: 5368.9 },
    { date_time: "2024-08-14 23:00", actual: 5578.0, predicted: 5198.3 },
    { date_time: "2024-08-15 00:00", actual: 5007.1, predicted: 5032.2 }
  ],

  error_by_building_type: [
    { building_type: "Apartment", mae: 118.42, rmse: 201.37, sample_count: 1200 },
    { building_type: "Commercial", mae: 176.85, rmse: 284.54, sample_count: 980 },
    { building_type: "University", mae: 201.16, rmse: 319.48, sample_count: 860 },
    { building_type: "Hospital", mae: 298.74, rmse: 455.12, sample_count: 790 },
    { building_type: "Hotel", mae: 252.31, rmse: 401.26, sample_count: 640 }
  ],

  kpi: {
    expected_energy_kwh: 148230.5,
    peak_prediction_kw: 6612.9,
    demand_response_saving_kw: 284.6,
    confidence: 92.4,
    building_type: "Hospital",
    dominant_variables: ["기온", "습도", "요일/시간대"]
  },

  llm_report: {
    title: "전력 소비 예측 성능 분석 보고서",
    summary:
      "초기 모델은 급격한 변동 구간에서 오차가 크게 발생했으나, 재학습 이후 전체 추세 적합도가 개선되었습니다.",
    performance_analysis:
      "재학습 전 RMSE는 512.4로 기준을 초과했으며, 재학습 후 RMSE는 359.85로 감소하여 성능이 개선되었습니다.",
    pattern_analysis:
      "재학습 전에는 실제값의 급락 구간을 충분히 따라가지 못했으나, 재학습 후 예측 곡선이 실제값에 더 근접하는 경향이 확인되었습니다.",
    building_type_analysis:
      "병원 유형에서 오차가 가장 크게 나타났으며, 아파트 유형은 상대적으로 안정적인 성능을 보였습니다.",
    strengths: [
      "재학습 후 전체 추세 반영 능력이 개선됨",
      "정상 구간에서 실제값과 예측값 차이가 감소함"
    ],
    weaknesses: [
      "급격한 하락 구간에서는 여전히 오차가 일부 남아 있음",
      "변동성이 큰 건물 유형에서는 오차가 상대적으로 큼"
    ],
    action:
      "재학습 결과를 검토한 후 승인 시 운영 모델로 반영할 수 있습니다."
  }
};

function resetSteps() {
  ["upload", "predict", "evaluate", "retrain", "report"].forEach((name) => {
    const step = document.getElementById(`step-${name}`);
    if (step) step.className = "step";
  });
}

function setStep(name, status) {
  const step = document.getElementById(`step-${name}`);
  if (step) step.className = `step ${status}`;
}

function destroyChartIfExists(instance) {
  if (instance) instance.destroy();
}

function renderSingleChart(canvasId, data, chartInstance, titleSuffix) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const ctx = canvas.getContext("2d");
  const labels = (data || []).map((row) => row.date_time);
  const actual = (data || []).map((row) => row.actual);
  const predicted = (data || []).map((row) => row.predicted);

  destroyChartIfExists(chartInstance);

  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual",
          data: actual,
          borderWidth: 2.5,
          tension: 0.35
        },
        {
          label: "Predicted",
          data: predicted,
          borderWidth: 2.5,
          tension: 0.35
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      normalized: true,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          position: "top"
        },
        title: {
          display: true,
          text: `Prediction ${titleSuffix}`
        }
      },
      scales: {
        x: {
          ticks: {
            autoSkip: true,
            maxTicksLimit: 6
          }
        },
        y: {
          beginAtZero: false
        }
      }
    }
  });
}

function renderCompareCharts(before, after) {
  chartBeforeInstance = renderSingleChart("chartBefore", before, chartBeforeInstance, "Before");
  chartAfterInstance = renderSingleChart("chartAfter", after, chartAfterInstance, "After");
}

function renderErrorTable(rows) {
  if (!errorTableBody) return;

  errorTableBody.innerHTML = "";

  if (!rows || rows.length === 0) {
    errorTableBody.innerHTML = `<tr><td colspan="4">결과가 없습니다.</td></tr>`;
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.building_type ?? "-"}</td>
      <td>${Number(row.mae ?? 0).toFixed(2)}</td>
      <td>${Number(row.rmse ?? 0).toFixed(2)}</td>
      <td>${row.sample_count ?? "-"}</td>
    `;
    errorTableBody.appendChild(tr);
  });
}

function renderKpiSection(kpi) {
  if (!kpi) return;

  if (kpiEnergy) kpiEnergy.textContent = `${Number(kpi.expected_energy_kwh ?? 0).toFixed(1)} kWh`;
  if (kpiPeak) kpiPeak.textContent = `${Number(kpi.peak_prediction_kw ?? 0).toFixed(1)} kW`;
  if (kpiSaving) kpiSaving.textContent = `${Number(kpi.demand_response_saving_kw ?? 0).toFixed(1)} kW`;
  if (kpiConfidence) kpiConfidence.textContent = `${Number(kpi.confidence ?? 0).toFixed(1)}%`;
  if (kpiBuildingType) kpiBuildingType.textContent = kpi.building_type ?? "-";
  if (kpiVariables) kpiVariables.textContent = (kpi.dominant_variables || []).join(", ") || "-";
}

function renderReport(report, meta) {
  const reportSection = document.getElementById("report-section");
  if (!reportSection) return;

  let retrainStatusText = "";
  if (meta?.retrain_executed) {
    retrainStatusText = meta?.model_replaced
      ? "사용자 승인 후 운영 모델 교체가 완료되었습니다."
      : "재학습이 완료되었으며 현재는 사용자 승인 대기 상태입니다.";
  } else {
    retrainStatusText = "현재 모델 성능이 기준 이내로 유지되어 재학습은 수행되지 않았습니다.";
  }

  reportSection.innerHTML = `
    <h2>LLM 분석 보고서</h2>
    <div class="report-document">
      <div class="report-document-header">
        <p class="report-meta">Analysis Report</p>
        <h3>${report?.title || "전력 소비 예측 성능 분석 보고서"}</h3>
        <p class="report-date">생성 시각: ${new Date().toLocaleString()}</p>
      </div>

      <div class="report-chapter">
        <h4>1. 요약</h4>
        <p>${report?.summary || "-"}</p>
      </div>

      <div class="report-chapter">
        <h4>2. 성능 분석</h4>
        <p>${report?.performance_analysis || "-"}</p>
      </div>

      <div class="report-chapter">
        <h4>3. 예측 패턴 분석</h4>
        <p>${report?.pattern_analysis || "-"}</p>
      </div>

      <div class="report-chapter">
        <h4>4. 건물 유형별 오차 분석</h4>
        <p>${report?.building_type_analysis || "-"}</p>
      </div>

      <div class="report-chapter">
        <h4>5. 주요 강점</h4>
        <ul>
          ${(report?.strengths || []).map((item) => `<li>${item}</li>`).join("") || "<li>-</li>"}
        </ul>
      </div>

      <div class="report-chapter">
        <h4>6. 주요 한계</h4>
        <ul>
          ${(report?.weaknesses || []).map((item) => `<li>${item}</li>`).join("") || "<li>-</li>"}
        </ul>
      </div>

      <div class="report-chapter">
        <h4>7. 재학습 상태</h4>
        <p>${retrainStatusText}</p>
      </div>

      <div class="report-chapter">
        <h4>8. 권장 조치</h4>
        <p>${report?.action || "-"}</p>
      </div>
    </div>
  `;
}

function closeUploadPanel() {
  if (uploadPanel) {
    uploadPanel.style.display = "none";
  }
  uploadPanelVisible = false;

  if (uploadBtn) {
    uploadBtn.textContent = "파일 업로드";
  }
}

function hideUploadToolbar() {
  if (uploadToolbar) {
    uploadToolbar.style.display = "none";
  }
}

function showApprovalSection(message) {
  if (!approvalSection) return;
  if (approvalMessage) approvalMessage.textContent = message;
  approvalSection.style.display = "flex";
}

function hideApprovalSection() {
  if (!approvalSection) return;
  approvalSection.style.display = "none";
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runMockAnalysis() {
  const data = JSON.parse(JSON.stringify(mockData));
  pendingMockResult = data;

  resetSteps();
  hideApprovalSection();

  setStep("upload", "active");
  uploadStatus.textContent = "파일 업로드 중...";
  await delay(600);
  setStep("upload", "done");

  setStep("predict", "active");
  uploadStatus.textContent = "예측 수행 중...";
  await delay(600);
  setStep("predict", "done");

  setStep("evaluate", "active");
  uploadStatus.textContent = "성능 평가 중...";
  await delay(600);
  setStep("evaluate", "done");

  if (data.retrain_required) {
    setStep("retrain", "warn");
    uploadStatus.textContent = "재학습 수행 중...";
    await delay(900);
    setStep("retrain", "done");
  } else {
    setStep("retrain", "done");
  }

  setStep("report", "active");
  uploadStatus.textContent = "보고서 생성 중...";
  await delay(500);
  setStep("report", "done");

  uploadStatus.textContent = "분석 완료";

  if (rmseValue) rmseValue.textContent = data.rmse_after ?? data.rmse ?? "-";
  if (retrainStatus) {
    retrainStatus.textContent = data.retrain_required ? "승인 대기" : "재학습 없음";
  }
  if (summaryMessage) summaryMessage.textContent = data.message;
  if (retrainMessage) {
    retrainMessage.textContent = data.retrain_required
      ? "재학습 완료 · 운영 반영 대기"
      : "재학습 없음";
  }

  renderCompareCharts(data.preview_before, data.preview_after);
  renderErrorTable(data.error_by_building_type);
  renderKpiSection(data.kpi);
  renderReport(data.llm_report, {
    retrain_executed: data.retrain_executed,
    model_replaced: false
  });

  closeUploadPanel();
  hideUploadToolbar();

  if (data.retrain_required) {
    showApprovalSection("재학습 결과를 확인했습니다. 운영 모델에 반영하시겠습니까?");
  }
}

function bindApprovalEvents() {
  if (approveBtn) {
    approveBtn.addEventListener("click", () => {
      if (!pendingMockResult) return;

      pendingMockResult.model_replaced = true;

      if (retrainStatus) retrainStatus.textContent = "모델 교체 완료";
      if (summaryMessage) {
        summaryMessage.textContent = "재학습 모델이 승인되어 운영 모델로 교체되었습니다.";
      }
      if (uploadStatus) uploadStatus.textContent = "운영 반영 완료";
      if (approvalMessage) approvalMessage.textContent = "운영 반영 승인이 완료되었습니다.";

      renderReport(pendingMockResult.llm_report, {
        retrain_executed: pendingMockResult.retrain_executed,
        model_replaced: true
      });
    });
  }

  if (rejectBtn) {
    rejectBtn.addEventListener("click", () => {
      if (!pendingMockResult) return;

      pendingMockResult.model_replaced = false;

      if (retrainStatus) retrainStatus.textContent = "운영 반영 보류";
      if (summaryMessage) {
        summaryMessage.textContent = "재학습 결과는 생성되었지만 운영 모델 교체는 보류되었습니다.";
      }
      if (uploadStatus) uploadStatus.textContent = "운영 반영 보류";
      if (approvalMessage) approvalMessage.textContent = "운영 반영이 보류되었습니다. 결과만 유지합니다.";

      renderReport(pendingMockResult.llm_report, {
        retrain_executed: pendingMockResult.retrain_executed,
        model_replaced: false
      });
    });
  }
}

function initUploadFlow() {
  if (uploadBtn) {
    uploadBtn.addEventListener("click", () => {
      uploadPanelVisible = !uploadPanelVisible;
      if (uploadPanel) {
        uploadPanel.style.display = uploadPanelVisible ? "flex" : "none";
      }
      uploadBtn.textContent = uploadPanelVisible ? "업로드 닫기" : "파일 업로드";
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      await runMockAnalysis();
    });
  }

  bindApprovalEvents();
}

initUploadFlow();