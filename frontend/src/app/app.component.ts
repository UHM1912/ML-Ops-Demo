import { Component, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PipelineService } from './pipeline.service';
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  activeTab: string = 'orchestrator';
  chatInput: string = '';
  chatMessages: Array<{ text: string; isUser: boolean; systemData?: any }> = [
    {
      text: 'Hello! I am your local MLOps AI Assistant. I can trigger actions in your local training pipeline using natural language.\n\nTry checking data drift, fetching recent runs, or retraining the classifier. Or click the quick actions underneath!',
      isUser: false
    }
  ];
  isTyping: boolean = false;

  // Prediction Form
  predictionForm = {
    Gender: 'Male',
    Age: 38,
    AnnualPremium: 2630.0,
    PastAccident: 'No',
    HasDrivingLicense: 1,
    Switch: 0
  };

  predResult = {
    show: false,
    text: '',
    class: 0,
    isError: false
  };

  // Health Status
  health = {
    api: 'dot-green',
    apiText: 'Online',
    mlflow: 'dot-green',
    mlflowText: 'Online',
    dvc: 'dot-green',
    dvcText: 'Configured',
    activeModel: 'Loaded',
    modelStyle: {
      background: 'var(--ml-bg)',
      color: '#115e59',
      borderColor: 'var(--ml-border)'
    }
  };

  // MLflow Runs
  runs: Array<any> = [];
  chartInstance: Chart | null = null;
  refreshingMLflow: boolean = false;
  showRefreshSuccess: boolean = false;

  // Drift Analysis
  drift = {
    status: '-',
    statusClass: '',
    features: '-',
    share: '-',
    total: '-',
    loading: false,
    loaded: false,
    error: ''
  };

  // CI/CD Simulator
  cicdLogs: Array<string> = [];
  cicdRunning: boolean = false;
  cicdStep: number = 0;

  private healthInterval: any;

  constructor(private pipelineService: PipelineService) {}

  ngOnInit() {
    this.runHealthCheck();
    this.healthInterval = setInterval(() => this.runHealthCheck(), 5000);
  }

  loadSampleProfile(type: 'likely' | 'unlikely') {
    if (type === 'likely') {
      this.predictionForm = {
        Gender: 'Female',
        Age: 57,
        AnnualPremium: 1444.40,
        PastAccident: 'Yes',
        HasDrivingLicense: 1,
        Switch: -1
      };
    } else {
      this.predictionForm = {
        Gender: 'Male',
        Age: 20,
        AnnualPremium: 1000.00,
        PastAccident: 'No',
        HasDrivingLicense: 1,
        Switch: 1
      };
    }
    this.predResult.show = false;
  }

  ngOnDestroy() {
    if (this.healthInterval) {
      clearInterval(this.healthInterval);
    }
    if (this.chartInstance) {
      this.chartInstance.destroy();
    }
  }

  switchTab(tabId: string) {
    this.activeTab = tabId;
    if (tabId === 'tab-mlflow') {
      setTimeout(() => this.loadMLflowRuns(), 50);
    }
  }

  sendQuickCommand(cmd: string) {
    this.chatInput = cmd;
    this.sendMessage();
  }

  sendMessage() {
    const text = this.chatInput.trim();
    if (!text) return;

    this.chatInput = '';
    this.chatMessages.push({ text, isUser: true });
    this.isTyping = true;

    this.pipelineService.sendMessage(text).subscribe({
      next: (data) => {
        this.isTyping = false;
        if (data.error) {
          this.chatMessages.push({ text: 'Error: ' + data.error, isUser: false });
        } else {
          this.chatMessages.push({ text: data.response, isUser: false, systemData: data });
          if (data.action === 'TRAIN') {
            this.runHealthCheck();
          }
        }
      },
      error: (err) => {
        this.isTyping = false;
        this.chatMessages.push({
          text: 'Failed to communicate with your local FastAPI server. Make sure it is running.',
          isUser: false
        });
      }
    });
  }

  runFormPrediction() {
    const payload = {
      ...this.predictionForm,
      Age: Number(this.predictionForm.Age),
      AnnualPremium: Number(this.predictionForm.AnnualPremium),
      HasDrivingLicense: Number(this.predictionForm.HasDrivingLicense),
      Switch: Number(this.predictionForm.Switch),
      RegionID: 28.0
    };

    this.pipelineService.predict(payload).subscribe({
      next: (data) => {
        this.predResult.show = true;
        if (data.error) {
          this.predResult.text = 'Error: ' + data.error;
          this.predResult.class = 0;
          this.predResult.isError = true;
        } else {
          this.predResult.isError = false;
          this.predResult.class = data.predicted_class;
          if (data.predicted_class === 1) {
            this.predResult.text = 'Class 1: Customer is LIKELY to buy insurance cross-sell.';
          } else {
            this.predResult.text = 'Class 0: Customer is UNLIKELY to buy insurance cross-sell.';
          }
        }
      },
      error: (err) => {
        alert('Failed to reach prediction API: ' + err.message);
      }
    });
  }

  loadMLflowRuns() {
    this.refreshingMLflow = true;
    this.pipelineService.getExperiments().subscribe({
      next: (data) => {
        this.refreshingMLflow = false;
        this.showRefreshSuccess = true;
        setTimeout(() => this.showRefreshSuccess = false, 1500);

        if (data.status !== 'success' || !data.latest_runs || data.latest_runs.length === 0) {
          this.runs = [];
          this.health.mlflow = 'dot-yellow';
          this.health.mlflowText = 'Inactive';
          return;
        }

        this.health.mlflow = 'dot-green';
        this.health.mlflowText = 'Online';
        this.runs = data.latest_runs;

        // Populate Chart
        const reversedRuns = [...data.latest_runs].reverse();
        const runNames = reversedRuns.map((r: any) => r.run_name || r.run_id.substring(0, 6));
        const accuracies = reversedRuns.map((r: any) => r.metrics.accuracy || 0.0);
        const rocScores = reversedRuns.map((r: any) => r.metrics.roc || 0.0);

        this.updateChart(runNames, accuracies, rocScores);
      },
      error: (err) => {
        this.refreshingMLflow = false;
        this.health.mlflow = 'dot-red';
        this.health.mlflowText = 'Offline';
      }
    });
  }

  updateChart(labels: string[], accs: number[], rocs: number[]) {
    const canvas = document.getElementById('mlflowChart') as HTMLCanvasElement;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (this.chartInstance) {
      this.chartInstance.destroy();
    }

    this.chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Accuracy',
            data: accs,
            borderColor: '#3ca88d',
            backgroundColor: 'rgba(60, 168, 141, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true
          },
          {
            label: 'ROC AUC',
            data: rocs,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#64748b', font: { family: 'Outfit' } }
          }
        },
        scales: {
          y: {
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: { color: '#64748b', font: { family: 'Outfit' } },
            min: 0.5,
            max: 1.0
          },
          x: {
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: { color: '#64748b', font: { family: 'Outfit' } }
          }
        }
      }
    });
  }

  triggerDriftCheck() {
    this.drift.loading = true;
    this.drift.loaded = false;
    this.drift.error = '';

    this.pipelineService.triggerDriftCheck().subscribe({
      next: (data) => {
        this.drift.loading = false;
        if (data.status === 'success') {
          const sum = data.summary;
          this.drift.status = sum.drift_detected ? 'DETECTED' : 'CLEAR';
          this.drift.statusClass = sum.drift_detected ? 'kpi-yellow' : 'kpi-green';
          this.drift.features = `${sum.number_of_drifted_features} / ${sum.total_features}`;
          this.drift.share = `${(sum.share_of_drifted_features * 100).toFixed(1)}%`;
          this.drift.total = sum.total_features;
          this.drift.loaded = true;
        } else {
          this.drift.error = data.error || data.message;
        }
      },
      error: (err) => {
        this.drift.loading = false;
        this.drift.error = 'Failed to execute Drift endpoint check: ' + err.message;
      }
    });
  }

  runCicdSimulation() {
    this.cicdRunning = true;
    this.cicdStep = 0;
    this.cicdLogs = [];
 
    const logs = [
      { text: '[1/6] git clone git@github.com:mlops/mlops-project.git...', delay: 200 },
      { text: "Cloning into 'mlops-project'...", delay: 500 },
      { text: "Checking out branch 'main'... OK", delay: 900 },
      { text: '[2/6] Setting up Python 3.10 and restoring pip cache...', delay: 1300 },
      { text: 'Successfully restored 45 pip packages.', delay: 1700 },
      { text: '[3/6] Running model tests and assertions...', delay: 2100 },
      { text: 'pytest tests/test_dataset.py::test_load_data ... [PASSED]', delay: 2600 },
      { text: 'pytest tests/test_clean.py::test_clean_data ... [PASSED]', delay: 3000 },
      { text: 'All 2 unit tests passed successfully.', delay: 3200 },
      { text: '[4/6] Verifying Data Version Control (DVC) checkout mappings...', delay: 3700 },
      { text: 'DVC Remote check: google-drive remote active.', delay: 4200 },
      { text: 'Verifying data files integrity check (data/train.csv.dvc)... OK', delay: 4600 },
      { text: '[5/6] Executing model build dry-run (python main.py)...', delay: 5100 },
      { text: 'Ingesting data -> Cleaning -> Smote preprocessing -> Training Model...', delay: 5700 },
      { text: 'Dry run completed. Accuracy logged: 0.8524. Runs saved in MLflow... [SUCCESS]', delay: 6300 },
      { text: '[6/6] Packaging prediction service into Docker Container...', delay: 6800 },
      { text: 'docker build -f ./dockerfile -t insurance-mlops-app:latest .', delay: 7200 },
      { text: 'Sending build context to Docker daemon... 25.4 MB', delay: 7800 },
      { text: 'Step 1/12: FROM python:3.10 ... Cache hit', delay: 8100 },
      { text: 'Step 8/12: COPY steps/ ./steps/ ... Done', delay: 8400 },
      { text: 'Successfully built image tagged: insurance-mlops-app:latest', delay: 8800 },
      { text: '🐳 CI/CD Pipeline Build SUCCESS. Image is ready for production deployment.', delay: 9200 }
    ];
 
    logs.forEach((log) => {
      setTimeout(() => {
        this.cicdLogs.push(log.text);
        if (log.text.includes('[1/6]')) {
          this.cicdStep = 1;
        } else if (log.text.includes('[3/6]')) {
          this.cicdStep = 2;
        } else if (log.text.includes('[4/6]')) {
          this.cicdStep = 3;
        } else if (log.text.includes('[5/6]')) {
          this.cicdStep = 4;
        } else if (log.text.includes('[6/6]')) {
          this.cicdStep = 5;
        } else if (log.text.includes('🐳')) {
          this.cicdStep = 6;
          this.cicdRunning = false;
        }
      }, log.delay);
    });
  }

  runHealthCheck() {
    this.pipelineService.checkHealth().subscribe({
      next: (data) => {
        if (data.status === 'ready') {
          this.health.api = 'dot-green';
          this.health.apiText = 'Online';

          const isModelLoaded = data.model_loaded;
          this.health.activeModel = isModelLoaded ? 'Loaded' : 'Not Loaded';
          this.health.modelStyle = {
            background: isModelLoaded ? 'var(--ml-bg)' : 'rgba(239, 68, 68, 0.15)',
            color: isModelLoaded ? '#115e59' : '#ef4444',
            borderColor: isModelLoaded ? 'var(--ml-border)' : 'rgba(239, 68, 68, 0.3)'
          };
        }
      },
      error: () => {
        this.health.api = 'dot-red';
        this.health.apiText = 'Offline';
      }
    });
  }

  formatParams(params: any): string {
    if (!params) return 'None';
    return Object.entries(params)
      .map(([k, v]) => `${k}:${v}`)
      .join(', ');
  }
}
