let sessionId = null;

async function api(url, options = {}) {
  const res = await fetch(url, options);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : `Error ${res.status}`);
  return body;
}

async function refreshHealth() {
  const h = await api("/health");
  document.getElementById("model-status").textContent = h.model_trained ? "Model trained ✓" : "Model not trained";
}

async function refreshReport() {
  if (!sessionId) return;
  const r = await api(`/sessions/${sessionId}/report`);
  document.getElementById("counts").textContent = `Present ${r.present} / ${r.total}`;
  const csv = document.getElementById("csv-link");
  csv.href = `/sessions/${sessionId}/report?format=csv`;
  csv.hidden = false;
  document.getElementById("end-btn").hidden = false;
  const rows = document.getElementById("report-body");
  rows.innerHTML = "";
  for (const s of r.students) {
    const tr = document.createElement("tr");
    for (const v of [s.roll_no, s.name, s.status, s.method || ""]) {
      const td = document.createElement("td");
      td.textContent = v;  // textContent, never innerHTML, for user-supplied names
      if (v === s.status) td.className = s.status;
      tr.appendChild(td);
    }
    const action = document.createElement("td");
    if (s.status === "absent") {
      const btn = document.createElement("button");
      btn.className = "secondary";
      btn.textContent = "Mark present";
      btn.onclick = async () => {
        await api(`/sessions/${sessionId}/attendance`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ roll_no: s.roll_no }),
        });
        refreshReport();
      };
      action.appendChild(btn);
    }
    tr.appendChild(action);
    rows.appendChild(tr);
  }
}

document.getElementById("session-form").onsubmit = async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target));
  try {
    const s = await api("/sessions", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    sessionId = s.id;
    document.getElementById("session-info").textContent = `Session ${s.id}: ${s.class_name} ${s.subject}`;
    refreshReport();
  } catch (err) { alert(err.message); }
};

document.getElementById("photo-form").onsubmit = async (e) => {
  e.preventDefault();
  const out = document.getElementById("photo-result");
  if (!sessionId) { out.textContent = "Start a session first."; return; }
  try {
    const r = await api(`/sessions/${sessionId}/recognize`, { method: "POST", body: new FormData(e.target) });
    const known = r.faces.filter((f) => f.roll_no).map((f) => f.roll_no + (f.marked ? " (marked)" : ""));
    out.textContent = `${r.faces_found} face(s) found. Recognised: ${known.join(", ") || "none"}`;
    refreshReport();
  } catch (err) { out.textContent = err.message; }
};

document.getElementById("end-btn").onclick = async () => {
  await api(`/sessions/${sessionId}/end`, { method: "POST" });
  document.getElementById("session-info").textContent += " (ended)";
};

refreshHealth();
