"""
Generate a standalone dashboard HTML with embedded JSON data.
This solves the browser security issue with fetch() on file:/// URLs.
"""

import json
import os

def generate_dashboard():
    """Generate dashboard.html with embedded data from simulation_results.json."""
    
    # Read simulation results
    results_path = os.path.join(os.path.dirname(__file__), 'simulation_results.json')
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # HTML template with embedded data
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sigma-C LLM Drift Demo - Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #050505;
            color: #ffffff;
            padding: 0;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }}
        
        /* Subtle grid background */
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 40px 20px;
            position: relative;
            z-index: 1;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 60px;
            padding: 60px 30px;
            background: rgba(10, 10, 10, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px; /* Rounded corners kept */
            backdrop-filter: blur(20px);
            position: relative;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        }}
        
        h1 {{
            font-size: 3.5em;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
            margin-bottom: 15px;
            text-transform: none; /* Removed uppercase for cleaner look */
        }}
        
        h1 span {{
            background: linear-gradient(135deg, #ffffff 0%, #888888 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            font-size: 1.2em;
            color: #888888;
            font-weight: 400;
            letter-spacing: 0.01em;
            margin-top: 10px;
        }}
        
        .key-insight {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 30px;
            margin: 40px 0;
            font-size: 1.2em;
            font-weight: 400;
            color: #e0e0e0;
            text-align: center;
            font-style: italic;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(700px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .chart-container {{
            background: rgba(15, 15, 15, 0.6);
            padding: 30px;
            border-radius: 20px; /* Rounded corners */
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}
        
        .chart-container:hover {{
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }}
        
        .chart-title {{
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #ffffff;
            font-weight: 600;
            letter-spacing: 0.01em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .chart-description {{
            color: #888888;
            margin-bottom: 25px;
            font-size: 0.95em;
            line-height: 1.6;
            font-weight: 400;
        }}
        
        .chart-description strong {{
            color: #ffffff;
            font-weight: 600;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 40px;
        }}
        
        .stat-card {{
            background: rgba(15, 15, 15, 0.6);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            border-color: rgba(255, 255, 255, 0.2);
            background: rgba(20, 20, 20, 0.8);
        }}
        
        .stat-label {{
            color: #666666;
            font-size: 0.85em;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }}
        
        .stat-value {{
            font-size: 2.8em;
            font-weight: 700;
            color: #ffffff;
            margin: 10px 0;
            letter-spacing: -0.03em;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 100px;
            font-size: 0.8em;
            font-weight: 600;
            margin-top: 15px;
            letter-spacing: 0.02em;
        }}
        
        .status-stable {{
            background: #ffffff;
            color: #000000;
        }}
        
        .status-collapsed {{
            background: #333333;
            color: #ffffff;
            border: 1px solid #555555;
        }}
        
        canvas {{
            max-height: 400px;
        }}
        
        footer {{
            text-align: center;
            margin-top: 80px;
            color: #444;
            font-size: 0.9em;
            padding-bottom: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span>Sigma-C Framework</span></h1>
            <div class="subtitle">LLM Model Drift Detection & Active Control - Live Demonstration</div>
        </header>
        
        <div class="key-insight">
            "Traditional monitoring tells you your system is dead. Sigma-C tells you it's dying — and gives you the tools to save it."
        </div>
        
        <div id="content">
            <div class="chart-grid">
                <div class="chart-container">
                    <div class="chart-title">📊 Sentiment Score (Performance Metric)</div>
                    <div class="chart-description">
                        <strong>What you see:</strong> This graph tracks the output quality (Sentiment) of both systems over time. The <span style="color:#ff4444">Red Line (Baseline)</span> shows the traditional system, the <span style="color:#44ff44">Green Line (Sigma-C)</span> the protected system. Note: Both appear stable until Minute 20+, when the baseline suddenly collapses, while Sigma-C remains stable.
                    </div>
                    <canvas id="sentimentChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">⚡ Sigma-C Score (Susceptibility)</div>
                    <div class="chart-description">
                        <strong>What you see:</strong> The <span style="color:#ffaa00">Orange Line</span> shows the Sigma-C score (System Susceptibility). When this drops below the <span style="color:#ff4444">Red Dashed Line (Threshold 0.7)</span>, the system is critical. <strong>Important:</strong> Sigma-C detects the problem at <strong>Minute 19</strong> — long before performance degrades! Low Sigma-C = High Instability = Early Warning.
                    </div>
                    <canvas id="sigmaCChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">🌡️ Temperature (Control Parameter)</div>
                    <div class="chart-description">
                        <strong>What you see:</strong> The <span style="color:#888888">Gray Line (Baseline)</span> remains constant at 0.8. The <span style="color:#44aaff">Blue Line (Sigma-C)</span> shows how the Active Control System automatically reduces temperature (0.8 → 0.12) as soon as Sigma-C triggers an alert. <strong>Effect:</strong> Lower temperature = more deterministic = more stable. This is the automatic rescue action!
                    </div>
                    <canvas id="temperatureChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">☢️ Input Toxicity (Attack Vector)</div>
                    <div class="chart-description">
                        <strong>What you see:</strong> The <span style="color:#cc44ff">Purple Line</span> shows the gradual increase in input toxicity (0.1 → 0.77). This simulates a "drift attack" — e.g., increasingly toxic/confusing user inputs or distribution shift. This is the stressor testing both systems.
                    </div>
                    <canvas id="toxicityChart"></canvas>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Baseline Final Sentiment</div>
                    <div class="stat-value" id="baselineSentiment">-</div>
                    <div class="status-badge" id="baselineStatus">-</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Sigma-C Final Sentiment</div>
                    <div class="stat-value" id="sigmacSentiment">-</div>
                    <div class="status-badge" id="sigmacStatus">-</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Early Detection (Minutes Ahead)</div>
                    <div class="stat-value" id="earlyDetection">-</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Final Temperature (Sigma-C)</div>
                    <div class="stat-value" id="finalTemp">-</div>
                </div>
            </div>
            
            <footer>
                &copy; 2025 ForgottenForge.xyz - Sigma-C Framework Demo
            </footer>
        </div>
    </div>
    
    <script>
        // Embedded data from simulation
        const data = {json.dumps(data, indent=2)};
        
        const baseline = data.baseline;
        const sigmac = data.sigma_c_protected;
        const minutes = baseline.map(d => d.minute);
        
        // Chart.js default config for ForgottenForge theme
        Chart.defaults.color = '#666666';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
        Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        Chart.defaults.font.size = 11;
        
        // Sentiment Chart
        new Chart(document.getElementById('sentimentChart'), {{
            type: 'line',
            data: {{
                labels: minutes,
                datasets: [
                    {{
                        label: 'Baseline (Traditional)',
                        data: baseline.map(d => d.sentiment),
                        borderColor: '#ff4444',
                        backgroundColor: 'rgba(255, 68, 68, 0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }},
                    {{
                        label: 'Sigma-C Protected',
                        data: sigmac.map(d => d.sentiment),
                        borderColor: '#44ff44',
                        backgroundColor: 'rgba(68, 255, 68, 0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ 
                        display: true,
                        labels: {{
                            color: '#e0e0e0',
                            font: {{ size: 12 }},
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        title: {{ 
                            display: true, 
                            text: 'Sentiment Score',
                            color: '#666666'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)'
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }},
                    x: {{
                        title: {{ 
                            display: true, 
                            text: 'Time (Minutes)',
                            color: '#666666'
                        }},
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }}
                }}
            }}
        }});
        
        // Sigma-C Chart
        new Chart(document.getElementById('sigmaCChart'), {{
            type: 'line',
            data: {{
                labels: minutes,
                datasets: [
                    {{
                        label: 'Sigma-C Score',
                        data: sigmac.map(d => d.sigma_c),
                        borderColor: '#ffaa00',
                        backgroundColor: 'rgba(255, 170, 0, 0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }},
                    {{
                        label: 'Critical Threshold',
                        data: Array(minutes.length).fill(0.7),
                        borderColor: '#ff4444',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ 
                        display: true,
                        labels: {{
                            color: '#e0e0e0',
                            font: {{ size: 12 }},
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        title: {{ 
                            display: true, 
                            text: 'Sigma-C (Lower = More Critical)',
                            color: '#666666'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)'
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }},
                    x: {{
                        title: {{ 
                            display: true, 
                            text: 'Time (Minutes)',
                            color: '#666666'
                        }},
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }}
                }}
            }}
        }});
        
        // Temperature Chart
        new Chart(document.getElementById('temperatureChart'), {{
            type: 'line',
            data: {{
                labels: minutes,
                datasets: [
                    {{
                        label: 'Baseline Temperature',
                        data: baseline.map(d => d.temperature),
                        borderColor: '#666666',
                        backgroundColor: 'rgba(102, 102, 102, 0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }},
                    {{
                        label: 'Sigma-C Controlled Temperature',
                        data: sigmac.map(d => d.temperature),
                        borderColor: '#44aaff',
                        backgroundColor: 'rgba(68, 170, 255, 0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ 
                        display: true,
                        labels: {{
                            color: '#e0e0e0',
                            font: {{ size: 12 }},
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        title: {{ 
                            display: true, 
                            text: 'Temperature',
                            color: '#666666'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)'
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }},
                    x: {{
                        title: {{ 
                            display: true, 
                            text: 'Time (Minutes)',
                            color: '#666666'
                        }},
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }}
                }}
            }}
        }});
        
        // Toxicity Chart
        new Chart(document.getElementById('toxicityChart'), {{
            type: 'line',
            data: {{
                labels: minutes,
                datasets: [
                    {{
                        label: 'Input Toxicity',
                        data: baseline.map(d => d.toxicity),
                        borderColor: '#cc44ff',
                        backgroundColor: 'rgba(204, 68, 255, 0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ 
                        display: true,
                        labels: {{
                            color: '#e0e0e0',
                            font: {{ size: 12 }},
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        title: {{ 
                            display: true, 
                            text: 'Toxicity Level',
                            color: '#666666'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)'
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }},
                    x: {{
                        title: {{ 
                            display: true, 
                            text: 'Time (Minutes)',
                            color: '#666666'
                        }},
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            color: '#666666'
                        }}
                    }}
                }}
            }}
        }});
        
        // Update stats
        const baselineFinal = baseline[baseline.length - 1];
        const sigmacFinal = sigmac[sigmac.length - 1];
        
        document.getElementById('baselineSentiment').textContent = baselineFinal.sentiment.toFixed(2);
        document.getElementById('sigmacSentiment').textContent = sigmacFinal.sentiment.toFixed(2);
        document.getElementById('finalTemp').textContent = sigmacFinal.temperature.toFixed(2);
        
        const baselineStatusEl = document.getElementById('baselineStatus');
        baselineStatusEl.textContent = baselineFinal.sentiment < 0.5 ? 'COLLAPSED' : 'OK';
        baselineStatusEl.className = 'status-badge ' + (baselineFinal.sentiment < 0.5 ? 'status-collapsed' : 'status-stable');
        
        const sigmacStatusEl = document.getElementById('sigmacStatus');
        sigmacStatusEl.textContent = sigmacFinal.sentiment > 0.5 ? 'STABLE' : 'DEGRADED';
        sigmacStatusEl.className = 'status-badge ' + (sigmacFinal.sentiment > 0.5 ? 'status-stable' : 'status-collapsed');
        
        // Calculate early detection
        // Find when Sigma-C first alerted (alert == 1)
        const firstSigmaCAlert = sigmac.findIndex(d => d.alert == 1);
        // Find when baseline first alerted (alert == 1) 
        const firstBaselineAlert = baseline.findIndex(d => d.alert == 1);
        
        // Calculate how many minutes earlier Sigma-C detected the problem
        const earlyDetectionMinutes = (firstBaselineAlert >= 0 && firstSigmaCAlert >= 0) 
            ? (firstBaselineAlert - firstSigmaCAlert) 
            : 'N/A';
        
        document.getElementById('earlyDetection').textContent = 
            (typeof earlyDetectionMinutes === 'number' && earlyDetectionMinutes > 0) 
                ? earlyDetectionMinutes + ' Min' 
                : 'N/A';
    </script>
</body>
</html>
"""
    
    # Write standalone dashboard
    output_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Generated standalone dashboard: {output_path}")

if __name__ == "__main__":
    generate_dashboard()
