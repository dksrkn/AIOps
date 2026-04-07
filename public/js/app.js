const uploadBtn = document.getElementById("uploadBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const uploadPanel = document.getElementById("uploadPanel");
const csvFileInput = document.getElementById("csvFile");
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
let latestResult = null;

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
  const labels = (data || []).map((r) => r.date_time);
  const actual = (data || []).map((r) => r.actual);
  const predicted = (data || []).map((r) => r.predicted);

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

function renderKpi(data) {
  const kpi = data?.kpi || {};

  if (kpiEnergy) {
    kpiEnergy.textContent =
      kpi.expected_energy_kwh != null ? `${Number(kpi.expected_energy_kwh).toFixed(1)} kWh` : "-";
  }
  if (kpiPeak) {
    kpiPeak.textContent =
      kpi.peak_prediction_kw != null ? `${Number(kpi.peak_prediction_kw).toFixed(1)} kW` : "-";
  }
  if (kpiSaving) {
    kpiSaving.textContent =
      kpi.demand_response_saving_kw != null
        ? `${Number(kpi.demand_response_saving_kw).toFixed(1)} kW`
        : "-";
  }
  if (kpiConfidence) {
    kpiConfidence.textContent =
      kpi.confidence != null ? `${Number(kpi.confidence).toFixed(1)}%` : "-";
  }
  if (kpiBuildingType) {
    kpiBuildingType.textContent = kpi.building_type ?? "-";
  }
  if (kpiVariables) {
    kpiVariables.textContent = (kpi.dominant_variables || []).join(", ") || "-";
  }
}

function renderReport(report, meta) {
  const reportSection = document.getElementById("report-section");
  if (!reportSection) return;

  let retrainStatusText = "";
  if (meta?.retrain_executed) {
    retrainStatusText = meta?.model_replaced
      ? "운영 반영이 완료되었습니다."
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
        <h4>2. 성능 분석 변화</h4>
        <p>${report?.performance_analysis || "-"}</p>
      </div>

      <div class="report-chapter">
        <h4>3. 건물 유형별 오차 분석</h4>
        <p>${report?.building_type_analysis || "-"}</p>
      </div>

      <div class="report-chapter">
        <h4>4. 재학습 상태</h4>
        <p>${retrainStatusText}</p>
      </div>

      <div class="report-chapter">
        <h4>5. 권장 조치</h4>
        <p>${report?.action || "-"}</p>
      </div>
    </div>
  `;
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

function closeUploadPanel() {
  if (uploadPanel) uploadPanel.style.display = "none";
  uploadPanelVisible = false;
  if (uploadBtn) uploadBtn.textContent = "파일 업로드";
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/upload", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    let errorMessage = "업로드 중 오류가 발생했습니다.";
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch (error) {}
    throw new Error(errorMessage);
  }

  return await response.json();
}

async function approveModel() {
  const response = await fetch("http://127.0.0.1:8000/approve", {
    method: "POST"
  });

  if (!response.ok) {
    let errorMessage = "운영 반영 승인 중 오류가 발생했습니다.";
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch (error) {}
    throw new Error(errorMessage);
  }

  return await response.json();
}

function fillResult(data) {
  latestResult = data;

  if (rmseValue) rmseValue.textContent = data.rmse_after ?? data.rmse ?? "-";
  if (retrainStatus) {
    retrainStatus.textContent = data.retrain_required ? "승인 대기" : "재학습 없음";
  }
  if (summaryMessage) summaryMessage.textContent = data.message || "-";
  if (retrainMessage) {
    retrainMessage.textContent = data.retrain_required
      ? "재학습 완료 · 운영 반영 대기"
      : "재학습 없음";
  }

  renderCompareCharts(data.preview_before || [], data.preview_after || []);
  renderErrorTable(data.error_by_building_type || []);
  renderKpi(data);

  renderReport(data.llm_report, {
    retrain_executed: data.retrain_executed,
    model_replaced: false
  });
}

async function runBackendAnalysis() {
  const file = csvFileInput?.files?.[0];

  if (!file) {
    alert("CSV 파일을 선택하세요.");
    return;
  }

  try {
    if (analyzeBtn) analyzeBtn.disabled = true;

    resetSteps();
    hideApprovalSection();

    setStep("upload", "active");
    uploadStatus.textContent = "파일 업로드 중...";

    const data = await uploadFile(file);

    setStep("upload", "done");
    setStep("predict", "done");
    setStep("evaluate", "done");

    if (data.retrain_required) {
      setStep("retrain", data.retrain_executed ? "done" : "warn");
    } else {
      setStep("retrain", "done");
    }

    setStep("report", "done");
    uploadStatus.textContent = "분석 완료";

    fillResult(data);
    closeUploadPanel();

    if (data.retrain_required && data.retrain_executed) {
      showApprovalSection("재학습 결과를 확인했습니다. 운영 모델에 반영하시겠습니까?");
    }
  } catch (error) {
    console.error(error);
    uploadStatus.textContent = "오류 발생";
    if (summaryMessage) {
      summaryMessage.textContent = error.message || "알 수 없는 오류가 발생했습니다.";
    }
    if (retrainMessage) {
      retrainMessage.textContent = "백엔드 응답을 확인해주세요.";
    }
    setStep("upload", "warn");
  } finally {
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

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
    await runBackendAnalysis();
  });
}

if (approveBtn) {
  approveBtn.addEventListener("click", async () => {
    try {
      await approveModel();

      if (retrainStatus) retrainStatus.textContent = "모델 교체 완료";
      if (summaryMessage) {
        summaryMessage.textContent = "재학습 모델이 승인되어 운영 모델로 교체되었습니다.";
      }
      if (uploadStatus) uploadStatus.textContent = "운영 반영 완료";
      if (approvalMessage) approvalMessage.textContent = "운영 반영 승인이 완료되었습니다.";

      renderReport(latestResult?.llm_report, {
        retrain_executed: latestResult?.retrain_executed,
        model_replaced: true
      });

      hideApprovalSection();
    } catch (error) {
      console.error(error);
      if (approvalMessage) {
        approvalMessage.textContent = error.message || "운영 반영 승인에 실패했습니다.";
      }
    }
  });
}

if (rejectBtn) {
  rejectBtn.addEventListener("click", () => {
    if (retrainStatus) retrainStatus.textContent = "운영 반영 보류";
    if (summaryMessage) {
      summaryMessage.textContent = "재학습 결과는 생성되었지만 운영 모델 교체는 보류되었습니다.";
    }
    if (uploadStatus) uploadStatus.textContent = "운영 반영 보류";
    if (approvalMessage) approvalMessage.textContent = "운영 반영이 보류되었습니다. 결과만 유지합니다.";

    renderReport(latestResult?.llm_report, {
      retrain_executed: latestResult?.retrain_executed,
      model_replaced: false
    });
  });
}