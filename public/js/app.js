const uploadBtn = document.getElementById("uploadBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const uploadPanel = document.getElementById("uploadPanel");
const csvFileInput = document.getElementById("csvFile");
const uploadStatus = document.getElementById("uploadStatus");

const rmseBeforeValue = document.getElementById("rmseBeforeValue");
const rmseAfterValue = document.getElementById("rmseAfterValue");
const rmseDeltaValue = document.getElementById("rmseDeltaValue");
const retrainStatus = document.getElementById("retrainStatus");
const candidateStatusMeta = document.getElementById("candidateStatusMeta");
const summaryMessage = document.getElementById("summaryMessage");
const retrainMessage = document.getElementById("retrainMessage");

const errorTableBody = document.getElementById("errorTableBody");

const approvalSection = document.getElementById("approvalSection");
const approvalMessage = document.getElementById("approvalMessage");
const approveBtn = document.getElementById("approveBtn");
const rejectBtn = document.getElementById("rejectBtn");
const API_BASE_URL = window.location.origin;

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
let approvalMode = null;
let retrainPollTimer = null;

function formatNumber(value, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return numeric.toLocaleString("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

function formatSignedDelta(value, digits = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${formatNumber(numeric, digits)}`;
}

function getCandidateStatusText(data) {
  if (data.retrain_required && data.retrain_executed && data.model_replaced) {
    return "운영 반영 완료";
  }
  if (data.retrain_required && data.retrain_executed) {
    return "후보 모델 생성됨";
  }
  if (data.retrain_required) {
    return "재학습 필요";
  }
  return "후보 모델 없음";
}

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

function formatDateLabel(value) {
  if (!value) return "-";

  const text = String(value);
  if (text.includes("T")) {
    return text.split("T")[0];
  }

  if (text.includes(" ")) {
    return text.split(" ")[0];
  }

  if (/^\d{8}/.test(text)) {
    return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
  }

  return text;
}

function aggregateChartSeriesByDateTime(data, includePredicted = true) {
  const rows = Array.isArray(data) ? data : [];
  const grouped = new Map();

  rows.forEach((row) => {
    if (!row?.date_time) return;

    const key = row.date_time;
    if (!grouped.has(key)) {
      grouped.set(key, {
        date_time: key,
        actual_sum: 0,
        actual_count: 0,
        predicted_sum: 0,
        predicted_count: 0
      });
    }

    const item = grouped.get(key);
    const actual = Number(row.actual);
    const predicted = Number(row.predicted);

    if (Number.isFinite(actual)) {
      item.actual_sum += actual;
      item.actual_count += 1;
    }

    if (includePredicted && Number.isFinite(predicted)) {
      item.predicted_sum += predicted;
      item.predicted_count += 1;
    }
  });

  return Array.from(grouped.values())
    .sort((a, b) => String(a.date_time).localeCompare(String(b.date_time)))
    .map((item) => ({
      date_time: item.date_time,
      actual: item.actual_count ? item.actual_sum / item.actual_count : null,
      predicted:
        includePredicted && item.predicted_count
          ? item.predicted_sum / item.predicted_count
          : null
    }));
}

function renderSingleChart(canvasId, data, chartInstance, titleSuffix) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const ctx = canvas.getContext("2d");
  const averagedRows = aggregateChartSeriesByDateTime(data, true);
  const labels = averagedRows.map((r) => r.date_time);
  const actual = averagedRows.map((r) => r.actual);
  const predicted = averagedRows.map((r) => r.predicted);

  destroyChartIfExists(chartInstance);

  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual",
          data: actual,
          borderColor: "#0f172a",
          backgroundColor: "rgba(15, 23, 42, 0.12)",
          borderWidth: 3,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4
        },
        {
          label: "Predicted",
          data: predicted,
          borderColor: "#0f766e",
          backgroundColor: "rgba(15, 118, 110, 0.16)",
          borderWidth: 3,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4
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
          position: "top",
          labels: {
            color: "#334155",
            boxWidth: 14,
            usePointStyle: true
          }
        },
        title: {
          display: true,
          text: `Prediction ${titleSuffix}`,
          color: "#0f172a"
        }
      },
      scales: {
        x: {
          grid: {
            color: "rgba(148, 163, 184, 0.22)"
          },
          border: {
            color: "rgba(148, 163, 184, 0.28)"
          },
          ticks: {
            color: "#475569",
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 6,
            callback(value, index) {
              const label = this.getLabelForValue(value);
              const current = formatDateLabel(label);
              const previous = index > 0 ? formatDateLabel(labels[index - 1]) : null;
              return current !== previous ? current : "";
            }
          },
          title: {
            display: true,
            text: "Datetime",
            color: "#64748b"
          }
        },
        y: {
          grid: {
            color: "rgba(148, 163, 184, 0.22)"
          },
          border: {
            color: "rgba(148, 163, 184, 0.28)"
          },
          ticks: {
            color: "#475569",
            callback(value) {
              return Number(value).toLocaleString("ko-KR");
            }
          }
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
      kpi.expected_energy_kwh != null ? `${formatNumber(kpi.expected_energy_kwh)} kWh` : "-";
  }
  if (kpiPeak) {
    kpiPeak.textContent =
      kpi.peak_prediction_kw != null ? `${formatNumber(kpi.peak_prediction_kw)} kW` : "-";
  }
  if (kpiSaving) {
    kpiSaving.textContent =
      kpi.demand_response_saving_kw != null
        ? `${formatNumber(kpi.demand_response_saving_kw)} kW`
        : "-";
  }
  if (kpiConfidence) {
    kpiConfidence.textContent =
      kpi.confidence != null ? `${formatNumber(kpi.confidence)}%` : "-";
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
  } else if (meta?.retrain_required) {
    retrainStatusText = "재학습이 필요하며, 현재는 사용자에게 실행 여부를 묻는 단계입니다.";
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
  approvalMode = null;
  approvalSection.style.display = "none";
}

function closeUploadPanel() {
  if (uploadPanel) uploadPanel.style.display = "none";
  uploadPanelVisible = false;
  if (uploadBtn) uploadBtn.textContent = "파일 업로드";
}

async function uploadFile(file, autoRetrain = false) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("auto_retrain", String(autoRetrain));

  const response = await fetch(`${API_BASE_URL}/upload`, {
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
  const response = await fetch(`${API_BASE_URL}/approve`, {
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

async function startRetrain() {
  const response = await fetch(`${API_BASE_URL}/retrain`, {
    method: "POST"
  });

  if (!response.ok) {
    let errorMessage = "재학습 시작 중 오류가 발생했습니다.";
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorData.message || errorMessage;
    } catch (error) {}
    throw new Error(errorMessage);
  }

  return await response.json();
}

async function fetchRetrainStatus() {
  const response = await fetch(`${API_BASE_URL}/retrain-status`);

  if (!response.ok) {
    throw new Error("재학습 상태를 확인할 수 없습니다.");
  }

  return await response.json();
}

function stopRetrainPolling() {
  if (retrainPollTimer) {
    clearTimeout(retrainPollTimer);
    retrainPollTimer = null;
  }
}

function formatRetrainStage(status) {
  const stageMap = {
    idle: "대기 중",
    preparing: "준비 중",
    preprocessing: "전처리 중",
    initial_training_40_20: "1차 학습 중",
    retraining_uploaded_window: "업로드 데이터 반영 재학습 중",
    retraining_60_20: "2차 학습 중",
    retraining_80_20: "3차 학습 중",
    final_model_save: "모델 저장 중",
    completed: "완료",
    failed: "실패"
  };

  return stageMap[status?.stage] || status?.stage || "진행 중";
}

async function pollRetrainUntilDone() {
  stopRetrainPolling();

  async function pollOnce() {
    try {
      const status = await fetchRetrainStatus();
      const progress = Number(status.progress_pct || 0).toFixed(0);
      const stageText = formatRetrainStage(status);

      if (uploadStatus) {
        uploadStatus.textContent = `재학습 ${progress}%`;
      }
      if (summaryMessage) {
        summaryMessage.textContent = `재학습 진행 중입니다. 현재 단계: ${stageText} (${progress}%)`;
      }
      if (retrainMessage) {
        retrainMessage.textContent = `재학습 진행 중 · ${stageText}`;
      }

      if (status.running) {
        retrainPollTimer = setTimeout(pollOnce, 2000);
        return;
      }

      stopRetrainPolling();

      if (status.stage === "completed") {
        if (status.last_result) {
          latestResult = {
            ...(latestResult || {}),
            ...status.last_result,
            retrain_required: true,
            retrain_executed: true,
            model_replaced: Boolean(status.last_result.model_replaced)
          };
          fillResult(latestResult);
        }

        setStep("retrain", "done");
        if (retrainStatus) retrainStatus.textContent = "후보 모델 생성 완료";
        if (candidateStatusMeta) candidateStatusMeta.textContent = "승인 전 · current model 유지";
        if (retrainMessage) retrainMessage.textContent = "재학습이 완료되었습니다. 아직 운영 모델은 교체되지 않았습니다.";
        if (uploadStatus) uploadStatus.textContent = "재학습 완료";
        if (summaryMessage) {
          summaryMessage.textContent = "재학습 후보 모델이 생성되었습니다.";
        }
        hideApprovalSection();
        return;
      }

      setStep("retrain", "warn");
      if (retrainStatus) retrainStatus.textContent = "재학습 실패";
      if (retrainMessage) retrainMessage.textContent = "재학습 중 오류가 발생했습니다.";
      if (uploadStatus) uploadStatus.textContent = "재학습 실패";
      if (summaryMessage) {
        summaryMessage.textContent = status.error || "재학습 중 오류가 발생했습니다.";
      }
      if (approvalMessage) {
        approvalMessage.textContent = status.error || "재학습 중 오류가 발생했습니다.";
      }
    } catch (error) {
      stopRetrainPolling();
      setStep("retrain", "warn");
      if (uploadStatus) uploadStatus.textContent = "상태 확인 실패";
      if (summaryMessage) {
        summaryMessage.textContent = error.message || "재학습 상태 확인에 실패했습니다.";
      }
    } finally {
      if (approveBtn) approveBtn.disabled = false;
      if (rejectBtn) rejectBtn.disabled = false;
    }
  }

  retrainPollTimer = setTimeout(pollOnce, 0);
}

function fillResult(data) {
  latestResult = data;

  const beforeRmse = data.rmse_before;
  const afterRmse = data.rmse_after ?? data.rmse;
  const rmseDelta =
    Number.isFinite(Number(beforeRmse)) && Number.isFinite(Number(afterRmse))
      ? Number(afterRmse) - Number(beforeRmse)
      : null;

  if (rmseBeforeValue) rmseBeforeValue.textContent = formatNumber(beforeRmse, 2);
  if (rmseAfterValue) rmseAfterValue.textContent = formatNumber(afterRmse, 2);
  if (rmseDeltaValue) rmseDeltaValue.textContent = formatSignedDelta(rmseDelta, 2);
  if (candidateStatusMeta) candidateStatusMeta.textContent = getCandidateStatusText(data);
  if (retrainStatus) {
    if (data.retrain_required && data.retrain_executed && data.model_replaced) {
      retrainStatus.textContent = "재학습 완료";
    } else if (data.retrain_required && data.retrain_executed) {
      retrainStatus.textContent = "승인 대기";
    } else if (data.retrain_required) {
      retrainStatus.textContent = "재학습 필요";
    } else {
      retrainStatus.textContent = "재학습 없음";
    }
  }
  if (summaryMessage) {
    summaryMessage.textContent = data.message || "-";
  }
  if (retrainMessage) {
    if (data.retrain_required && data.retrain_executed && data.model_replaced) {
      retrainMessage.textContent = "재학습이 완료되어 운영 모델에 반영되었습니다.";
    } else if (data.retrain_required && data.retrain_executed) {
      retrainMessage.textContent = "후보 모델 생성 완료 · 운영 반영 대기";
    } else if (data.retrain_required) {
      retrainMessage.textContent = "재학습 여부를 선택해주세요.";
    } else {
      retrainMessage.textContent = "재학습 없음";
    }
  }

  renderCompareCharts(data.preview_before || [], data.preview_after || []);
  renderErrorTable(data.error_by_building_type || []);
  renderKpi(data);

  renderReport(data.llm_report, {
    retrain_required: data.retrain_required,
    retrain_executed: data.retrain_executed,
    model_replaced: data.model_replaced
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

    const data = await uploadFile(file, false);

    setStep("upload", "done");
    setStep("predict", "done");
    setStep("evaluate", "done");

    if (data.retrain_required) {
      setStep("retrain", data.retrain_executed ? "done" : "active");
    } else {
      setStep("retrain", "done");
    }

    setStep("report", "done");
    uploadStatus.textContent = "분석 완료";

    fillResult(data);
    closeUploadPanel();

    if (data.retrain_required && !data.retrain_executed) {
      approvalMode = "retrain";
      if (approveBtn) approveBtn.textContent = "재학습 시작";
      if (rejectBtn) rejectBtn.textContent = "취소";
      showApprovalSection("RMSE가 임계치를 초과했습니다. 재학습을 시작하시겠습니까?");
    } else if (data.retrain_required && data.retrain_executed && !data.model_replaced) {
      approvalMode = "approve";
      if (approveBtn) approveBtn.textContent = "승인";
      if (rejectBtn) rejectBtn.textContent = "보류";
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
      if (approvalMode === "retrain") {
        if (approveBtn) approveBtn.disabled = true;
        if (rejectBtn) rejectBtn.disabled = true;
        if (uploadStatus) uploadStatus.textContent = "재학습 시작 중...";
        if (summaryMessage) {
          summaryMessage.textContent = "재학습을 백그라운드에서 시작합니다.";
        }
        if (retrainMessage) {
          retrainMessage.textContent = "후보 모델 생성을 시작했습니다.";
        }
        if (approvalMessage) {
          approvalMessage.textContent = "재학습이 시작되었습니다. 진행률을 확인하는 중입니다.";
        }

        await startRetrain();
        await pollRetrainUntilDone();

        return;
      }

      await approveModel();

      if (retrainStatus) retrainStatus.textContent = "모델 교체 완료";
      if (candidateStatusMeta) candidateStatusMeta.textContent = "후보 모델 승인됨";
      if (summaryMessage) {
        summaryMessage.textContent = "재학습 모델이 승인되어 운영 모델로 교체되었습니다.";
      }
      if (uploadStatus) uploadStatus.textContent = "운영 반영 완료";
      if (approvalMessage) approvalMessage.textContent = "운영 반영 승인이 완료되었습니다.";

      renderReport(latestResult?.llm_report, {
        retrain_required: latestResult?.retrain_required,
        retrain_executed: latestResult?.retrain_executed,
        model_replaced: true
      });

      hideApprovalSection();
    } catch (error) {
      console.error(error);
      if (approvalMessage) {
        approvalMessage.textContent = error.message || "운영 반영 승인에 실패했습니다.";
      }
    } finally {
      if (approveBtn) approveBtn.disabled = false;
      if (rejectBtn) rejectBtn.disabled = false;
    }
  });
}

if (rejectBtn) {
  rejectBtn.addEventListener("click", () => {
    if (approvalMode === "retrain") {
      if (retrainStatus) retrainStatus.textContent = "재학습 보류";
      if (summaryMessage) {
        summaryMessage.textContent = "재학습이 필요하지만 사용자가 실행을 보류했습니다.";
      }
      if (uploadStatus) uploadStatus.textContent = "재학습 보류";
      if (retrainMessage) {
        retrainMessage.textContent = "재학습이 아직 실행되지 않았습니다.";
      }
      hideApprovalSection();
      return;
    }

      if (retrainStatus) retrainStatus.textContent = "운영 반영 보류";
      if (candidateStatusMeta) candidateStatusMeta.textContent = "후보 모델 유지";
    if (summaryMessage) {
      summaryMessage.textContent = "재학습 결과는 생성되었지만 운영 모델 교체는 보류되었습니다.";
    }
    if (uploadStatus) uploadStatus.textContent = "운영 반영 보류";
    if (approvalMessage) approvalMessage.textContent = "운영 반영이 보류되었습니다. 결과만 유지합니다.";

    renderReport(latestResult?.llm_report, {
      retrain_required: latestResult?.retrain_required,
      retrain_executed: latestResult?.retrain_executed,
      model_replaced: false
    });
  });
}
