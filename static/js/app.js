// SmartHR Dashboard Frontend Utilities

// Global toast system
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "toast";
  if (type === "error") {
    toast.style.borderLeftColor = "var(--accent-red)";
  } else if (type === "warning") {
    toast.style.borderLeftColor = "var(--accent-yellow)";
  } else {
    toast.style.borderLeftColor = "var(--accent-green)";
  }

  toast.innerHTML = `
    <span style="font-weight: 500;">${message}</span>
  `;

  container.appendChild(toast);

  // Auto-remove toast after 4s
  setTimeout(() => {
    toast.style.animation = "slideIn 0.3s ease-out reverse";
    setTimeout(() => {
      toast.remove();
    }, 300);
  }, 4000);
}

// Load stats values
async function loadDashboardStats() {
  try {
    const res = await fetch("/api/stats");
    const stats = await res.json();

    document.getElementById("stat-total-employees").innerText = stats.total_employees;
    document.getElementById("stat-registered-faces").innerText = stats.registered_faces;
    document.getElementById("stat-today-attendance").innerText = stats.today_attendance;
    document.getElementById("stat-system-accuracy").innerText = `${stats.system_accuracy.toFixed(1)}%`;
  } catch (err) {
    console.error("Error loading dashboard stats:", err);
  }
}

// Load recent activity log
async function loadRecentActivity() {
  try {
    const res = await fetch("/api/recent_activity");
    const data = await res.json();
    const tbody = document.getElementById("recent-activity-body");
    if (!tbody) return;

    tbody.innerHTML = "";

    if (data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No attendance records today.</td></tr>`;
      return;
    }

    data.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.employee_id}</td>
        <td>${row.name}</td>
        <td>${row.department}</td>
        <td>${row.date}</td>
        <td>${row.time}</td>
        <td><span class="badge ${row.status.toLowerCase() === 'verified' ? 'verified' : 'unknown'}">${row.status}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading recent activity:", err);
  }
}

// Initialize Charts
async function loadDashboardCharts() {
  try {
    const res = await fetch("/api/charts");
    const chartsData = await res.json();

    // Weekly Chart
    const ctxWeekly = document.getElementById("weeklyChart");
    if (ctxWeekly) {
      new Chart(ctxWeekly, {
        type: 'line',
        data: {
          labels: chartsData.weekly.labels,
          datasets: [{
            label: 'Attendance Count',
            data: chartsData.weekly.data,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            tension: 0.4,
            fill: true
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#9ca3af' }
            },
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#9ca3af' }
            }
          }
        }
      });
    }

    // Department Distribution Chart
    const ctxDept = document.getElementById("departmentChart");
    if (ctxDept) {
      new Chart(ctxDept, {
        type: 'doughnut',
        data: {
          labels: chartsData.departments.labels,
          datasets: [{
            data: chartsData.departments.data,
            backgroundColor: [
              '#3b82f6',
              '#8b5cf6',
              '#10b981',
              '#f59e0b',
              '#ef4444',
              '#ec4899'
            ],
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: '#9ca3af', boxWidth: 12 }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error("Error loading dashboard charts:", err);
  }
}
