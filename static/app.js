document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("image");
  const startButton = document.getElementById("start-camera");
  const stopButton = document.getElementById("stop-camera");
  const video = document.getElementById("camera-stream");
  const canvas = document.getElementById("camera-canvas");
  const status = document.getElementById("camera-status");
  const cameraPlaceholder = document.getElementById("camera-placeholder");
  const uploadResultView = document.getElementById("upload-result-view");
  const liveResultView = document.getElementById("live-result-view");
  const liveAnnotatedImage = document.getElementById("live-annotated-image");
  const livePreviewEmpty = document.getElementById("live-preview-empty");
  const liveCropImage = document.getElementById("live-crop-image");
  const liveCropEmpty = document.getElementById("live-crop-empty");
  const liveBestLabel = document.getElementById("live-best-label");
  const liveConfidence = document.getElementById("live-confidence");
  const liveDetectionNote = document.getElementById("live-detection-note");
  const liveMeta = document.getElementById("live-meta");
  const liveRankingList = document.getElementById("live-ranking-list");
  const modeButtons = Array.from(document.querySelectorAll("[data-mode-target]"));
  const modePanels = Array.from(document.querySelectorAll("[data-mode-panel]"));

  const requiredElements = {
    fileInput, startButton, stopButton, video, canvas, status, cameraPlaceholder,
    uploadResultView, liveResultView, liveAnnotatedImage, livePreviewEmpty,
    liveCropImage, liveCropEmpty, liveBestLabel, liveConfidence, liveDetectionNote,
    liveMeta, liveRankingList
  };
  
  let missing = false;
  for (const [name, el] of Object.entries(requiredElements)) {
    if (!el) {
      console.error("app.js missing element:", name);
      missing = true;
    }
  }
  if (modeButtons.length === 0) { console.error("Missing modeButtons"); missing = true; }
  if (modePanels.length === 0) { console.error("Missing modePanels"); missing = true; }

  if (missing) {
    console.error("app.js aborting due to missing elements.");
    return;
  }

  let activeMode = "upload";
  let stream = null;
  let analysisTimer = null;
  let analysisSession = 0;

  const setStatus = (message, tone = "idle") => {
    status.textContent = message;
    status.classList.remove("ready", "error");
    if (tone === "ready" || tone === "error") {
      status.classList.add(tone);
    }
  };

  const renderRanking = (predictions) => {
    if (!predictions.length) {
      liveRankingList.innerHTML = `
        <li class="ranking-placeholder">
          No live prediction yet.
        </li>
      `;
      return;
    }

    liveRankingList.innerHTML = predictions
      .map(
        (pred, index) => `
      <li class="ranking-row">
        <div class="ranking-meta">
          <span class="ranking-label">${index + 1}. ${pred.label}</span>
          <span class="ranking-value">${(pred.score * 100).toFixed(1)}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${pred.score * 100}%;"></div>
        </div>
        <div class="ranking-class-id">Class ID: ${pred.class_id}</div>
      </li>
        `
      )
      .join("");
  };

  const resetLiveResult = () => {
    liveBestLabel.textContent = "Waiting for webcam";
    liveConfidence.textContent = "Start the camera to begin real-time predictions.";
    liveDetectionNote.textContent =
      "The model will localize the sign and update the result automatically.";
    liveMeta.textContent = "No live frame analyzed yet.";
    liveAnnotatedImage.hidden = true;
    liveAnnotatedImage.removeAttribute("src");
    livePreviewEmpty.hidden = false;
    liveCropImage.hidden = true;
    liveCropImage.removeAttribute("src");
    liveCropEmpty.hidden = false;
    renderRanking([]);
  };

  const setMode = (mode) => {
    activeMode = mode;

    for (const button of modeButtons) {
      const isActive = button.dataset.modeTarget === mode;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    }

    for (const panel of modePanels) {
      const isActive = panel.dataset.modePanel === mode;
      panel.classList.toggle("is-active", isActive);
      panel.hidden = !isActive;
    }

    const isWebcamMode = mode === "webcam";
    uploadResultView.hidden = isWebcamMode;
    liveResultView.hidden = !isWebcamMode;

    if (!isWebcamMode) {
      stopCamera({ resetLiveState: false, keepStatusMessage: false });
    } else {
      resetLiveResult();
    }
  };

  const scheduleNextAnalysis = (callback, delayMs) => {
    if (analysisTimer !== null) {
      window.clearTimeout(analysisTimer);
    }
    analysisTimer = window.setTimeout(callback, delayMs);
  };

  const stopStream = () => {
    if (!stream) {
      return;
    }

    for (const track of stream.getTracks()) {
      track.stop();
    }
    stream = null;
    video.srcObject = null;
    video.classList.remove("is-active");
    cameraPlaceholder.hidden = false;
  };

  const stopCamera = ({ resetLiveState = false, keepStatusMessage = false } = {}) => {
    analysisSession += 1;
    if (analysisTimer !== null) {
      window.clearTimeout(analysisTimer);
      analysisTimer = null;
    }

    stopStream();
    startButton.disabled = false;
    stopButton.disabled = true;

    if (resetLiveState) {
      resetLiveResult();
    }

    if (!keepStatusMessage) {
      setStatus("Webcam idle. Start it to launch live detection.");
    }
  };

  const captureFrameDataUrl = () => {
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (!sourceWidth || !sourceHeight) {
      return null;
    }

    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / sourceWidth);
    const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
    const targetHeight = Math.max(1, Math.round(sourceHeight * scale));

    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, targetWidth, targetHeight);
    return canvas.toDataURL("image/jpeg", 0.78);
  };

  const updateLiveResult = (payload, latencyMs) => {
    const predictions = Array.isArray(payload.predictions) ? payload.predictions : [];
    const bestPrediction = predictions[0];

    if (bestPrediction) {
      liveBestLabel.textContent = bestPrediction.label;
      liveConfidence.textContent = `Confidence: ${bestPrediction.percentage.toFixed(2)}%`;
    } else {
      liveBestLabel.textContent = "No prediction";
      liveConfidence.textContent = "The model did not return a class for this frame.";
    }

    liveDetectionNote.textContent =
      payload.detection_message || "Live detection result updated.";

    if (payload.preview_url) {
      liveAnnotatedImage.src = payload.preview_url;
      liveAnnotatedImage.hidden = false;
      livePreviewEmpty.hidden = true;
    }

    if (payload.localized_preview_url) {
      liveCropImage.src = payload.localized_preview_url;
      liveCropImage.hidden = false;
      liveCropEmpty.hidden = true;
    } else {
      liveCropImage.hidden = true;
      liveCropImage.removeAttribute("src");
      liveCropEmpty.hidden = false;
    }

    const detectionSource = payload.detection_box ? payload.detection_box.source : "full image fallback";
    liveMeta.textContent =
      `Frame ${payload.image_width}x${payload.image_height} | ` +
      `Region: ${detectionSource} | ` +
      `Round trip: ${Math.round(latencyMs)} ms`;

    renderRanking(predictions);
  };

  const startAnalysisLoop = () => {
    analysisSession += 1;
    const currentSession = analysisSession;

    const analyzeCurrentFrame = async () => {
      if (currentSession !== analysisSession || !stream || activeMode !== "webcam") {
        return;
      }

      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        scheduleNextAnalysis(analyzeCurrentFrame, 300);
        return;
      }

      const imageData = captureFrameDataUrl();
      if (!imageData) {
        scheduleNextAnalysis(analyzeCurrentFrame, 300);
        return;
      }

      const requestStartedAt = performance.now();
      try {
        const response = await fetch("/api/analyze-frame", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ image: imageData })
        });

        const payload = await response.json();
        if (currentSession !== analysisSession) {
          return;
        }

        if (!response.ok) {
          throw new Error(payload.error || "Live analysis failed.");
        }

        updateLiveResult(payload, performance.now() - requestStartedAt);
        setStatus("Webcam active. Live detection is running.", "ready");
      } catch (error) {
        if (currentSession !== analysisSession) {
          return;
        }

        setStatus(error.message || "Live analysis failed.", "error");
      }

      if (currentSession === analysisSession && stream && activeMode === "webcam") {
        scheduleNextAnalysis(analyzeCurrentFrame, 700);
      }
    };

    analyzeCurrentFrame();
  };

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      setMode("upload");
    }
  });

  for (const button of modeButtons) {
    button.addEventListener("click", () => {
      const mode = button.dataset.modeTarget;
      if (!mode || mode === activeMode) {
        return;
      }

      setMode(mode);
    });
  }

  startButton.addEventListener("click", async () => {
    setMode("webcam");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("This browser does not support webcam access.", "error");
      return;
    }

    stopCamera({ resetLiveState: true, keepStatusMessage: true });
    startButton.disabled = true;
    stopButton.disabled = true;
    setStatus("Requesting webcam access...");

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      video.srcObject = stream;
      await video.play();
      video.classList.add("is-active");
      cameraPlaceholder.hidden = true;
      startButton.disabled = true;
      stopButton.disabled = false;
      setStatus("Webcam active. Preparing live detection...", "ready");
      startAnalysisLoop();
    } catch (error) {
      startButton.disabled = false;
      stopButton.disabled = true;
      stopStream();
      setStatus("Camera access failed. You can still use upload mode.", "error");
    }
  });

  stopButton.addEventListener("click", () => {
    stopCamera({ resetLiveState: false, keepStatusMessage: false });
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && activeMode === "webcam") {
      stopCamera({ resetLiveState: false, keepStatusMessage: true });
      setStatus("Webcam paused because the page is not visible.");
    }
  });

  window.addEventListener("beforeunload", () => {
    stopCamera({ resetLiveState: false, keepStatusMessage: true });
  });

  resetLiveResult();

  // ── Drag & Drop support ──────────────────────────────────────────────
  const dropZone = document.getElementById("drop-zone");
  const uploadForm = document.querySelector(".upload-form");

  if (dropZone && fileInput && uploadForm) {
    // Prevent default browser behavior for drag events on the whole page
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      document.body.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    // Visual feedback
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, () => {
        dropZone.classList.add("drop-zone--active");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove("drop-zone--active");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;

      // Case 1: File dropped directly (local file)
      if (dt.files && dt.files.length > 0) {
        fileInput.files = dt.files;
        uploadForm.submit();
        return;
      }

      // Case 2: Image dragged from a web page (URL)
      const imageUrl =
        dt.getData("text/uri-list") ||
        dt.getData("text/plain") ||
        dt.getData("URL");

      if (imageUrl && imageUrl.startsWith("http")) {
        // Fetch the image from the URL and convert to a File
        fetch(imageUrl)
          .then((res) => {
            if (!res.ok) throw new Error("Failed to fetch image");
            return res.blob();
          })
          .then((blob) => {
            const ext = blob.type.split("/")[1] || "png";
            const file = new File([blob], `dragged-image.${ext}`, {
              type: blob.type,
            });
            const container = new DataTransfer();
            container.items.add(file);
            fileInput.files = container.files;
            uploadForm.submit();
          })
          .catch((err) => {
            console.error("Drag-from-web failed:", err);
            alert(
              "Could not load that image. Try right-clicking the image, saving it to your computer, then dragging the saved file here."
            );
          });
        return;
      }

      // Case 3: HTML with embedded <img> tag (some browsers)
      const htmlData = dt.getData("text/html");
      if (htmlData) {
        const match = htmlData.match(/<img[^>]+src=["']([^"']+)["']/i);
        if (match && match[1] && match[1].startsWith("http")) {
          fetch(match[1])
            .then((res) => {
              if (!res.ok) throw new Error("Failed to fetch image");
              return res.blob();
            })
            .then((blob) => {
              const ext = blob.type.split("/")[1] || "png";
              const file = new File([blob], `dragged-image.${ext}`, {
                type: blob.type,
              });
              const container = new DataTransfer();
              container.items.add(file);
              fileInput.files = container.files;
              uploadForm.submit();
            })
            .catch((err) => {
              console.error("Drag-from-html failed:", err);
              alert(
                "Could not load that image. Try saving the image first, then dragging the file."
              );
            });
        }
      }
    });
  }
});
