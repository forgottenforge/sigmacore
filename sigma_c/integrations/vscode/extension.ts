// Sigma-C VSCode Extension
// =========================
// Copyright (c) 2025 ForgottenForge.xyz
//
// Provides real-time code criticality analysis in VSCode.
// Calls the sigma_c Python package via child_process for actual
// susceptibility computation instead of naive heuristics.
//
// SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

import * as vscode from 'vscode';
import { exec } from 'child_process';

let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;
let isMonitoring = false;
let analysisCache: Map<string, { sigma_c: number; kappa: number; timestamp: number }> = new Map();
let debounceTimer: NodeJS.Timeout | undefined;

const CACHE_TTL_MS = 30_000;  // 30 seconds
const DEBOUNCE_MS = 1_000;    // 1 second after save

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('Sigma-C');
    outputChannel.appendLine('Sigma-C extension activated');

    // Status bar
    statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = 'sigma_c: --';
    statusBarItem.tooltip = 'Click to analyze current file';
    statusBarItem.command = 'sigma-c.analyze';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Commands
    const analyzeCmd = vscode.commands.registerCommand(
        'sigma-c.analyze',
        () => analyzeCurrentFile()
    );
    const toggleCmd = vscode.commands.registerCommand(
        'sigma-c.toggleMonitoring',
        () => toggleMonitoring()
    );
    const showOutputCmd = vscode.commands.registerCommand(
        'sigma-c.showOutput',
        () => outputChannel.show()
    );
    context.subscriptions.push(analyzeCmd, toggleCmd, showOutputCmd);

    // Auto-analyze on save (debounced)
    vscode.workspace.onDidSaveTextDocument(document => {
        if (isMonitoring && document.languageId === 'python') {
            if (debounceTimer) {
                clearTimeout(debounceTimer);
            }
            debounceTimer = setTimeout(() => analyzeDocument(document), DEBOUNCE_MS);
        }
    });
}

async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showInformationMessage('No active editor');
        return;
    }
    await analyzeDocument(editor.document);
}

async function analyzeDocument(document: vscode.TextDocument) {
    const filePath = document.uri.fsPath;

    // Check cache
    const cached = analysisCache.get(filePath);
    if (cached && (Date.now() - cached.timestamp) < CACHE_TTL_MS) {
        updateStatusBar(cached.sigma_c, cached.kappa);
        return;
    }

    // Only analyze Python files
    if (!filePath.endsWith('.py')) {
        statusBarItem.text = 'sigma_c: N/A';
        statusBarItem.backgroundColor = undefined;
        return;
    }

    statusBarItem.text = 'sigma_c: ...';

    // Call actual sigma_c analysis via Python
    const pythonPath = vscode.workspace.getConfiguration('sigma-c').get<string>('pythonPath', 'python');
    const escapedPath = filePath.replace(/\\/g, '\\\\').replace(/'/g, "\\'");

    const script = `
import json
from sigma_c.monitoring.github_actions import CIAnalyzer
analyzer = CIAnalyzer()
result = analyzer.analyze_file('${escapedPath}')
print(json.dumps({
    'sigma_c': result['sigma_c'],
    'kappa': result['metrics'].get('max_complexity', 0) / 20.0,
    'max_complexity': result['metrics'].get('max_complexity', 0),
    'n_functions': result['metrics'].get('n_functions', 0),
    'max_depth': result['metrics'].get('max_nesting_depth', 0)
}))
`;

    exec(`${pythonPath} -c "${script.replace(/"/g, '\\"')}"`, { timeout: 10_000 }, (error, stdout, stderr) => {
        if (error) {
            outputChannel.appendLine(`Analysis error: ${error.message}`);
            // Fallback to basic heuristic if sigma_c is not installed
            const text = document.getText();
            const sigma_c = fallbackAnalysis(text);
            updateStatusBar(sigma_c, 0);
            return;
        }

        try {
            const result = JSON.parse(stdout.trim());
            const sigma_c = result.sigma_c;
            const kappa = result.kappa;

            // Cache result
            analysisCache.set(filePath, { sigma_c, kappa, timestamp: Date.now() });

            updateStatusBar(sigma_c, kappa);

            outputChannel.appendLine(
                `[${new Date().toLocaleTimeString()}] ${filePath}: ` +
                `sigma_c=${sigma_c.toFixed(3)}, ` +
                `complexity=${result.max_complexity}, ` +
                `functions=${result.n_functions}, ` +
                `depth=${result.max_depth}`
            );
        } catch (e) {
            outputChannel.appendLine(`Parse error: ${stdout}`);
            statusBarItem.text = 'sigma_c: err';
        }
    });
}

function updateStatusBar(sigma_c: number, kappa: number) {
    statusBarItem.text = `sigma_c: ${sigma_c.toFixed(3)}`;

    const threshold = vscode.workspace.getConfiguration('sigma-c').get<number>('threshold', 0.7);

    if (sigma_c > threshold) {
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        statusBarItem.tooltip = `Criticality ${sigma_c.toFixed(3)} exceeds threshold ${threshold}`;
    } else {
        statusBarItem.backgroundColor = undefined;
        statusBarItem.tooltip = `Criticality: ${sigma_c.toFixed(3)} | Complexity: ${kappa.toFixed(1)}`;
    }
}

function fallbackAnalysis(text: string): number {
    // Simple fallback when sigma_c Python package is not available
    const lines = text.split('\n');
    const nonEmpty = lines.filter(l => l.trim().length > 0).length;

    // Count branching statements
    let branches = 0;
    for (const line of lines) {
        const trimmed = line.trim();
        if (/^(if |elif |else:|for |while |except |with |try:)/.test(trimmed)) {
            branches++;
        }
    }

    const branchDensity = nonEmpty > 0 ? branches / nonEmpty : 0;
    return Math.min(1.0, branchDensity * 5);
}

function toggleMonitoring() {
    isMonitoring = !isMonitoring;
    vscode.window.showInformationMessage(
        `Sigma-C monitoring ${isMonitoring ? 'enabled' : 'disabled'}`
    );

    if (isMonitoring) {
        // Analyze current file immediately
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            analyzeDocument(editor.document);
        }
    }
}

export function deactivate() {
    if (statusBarItem) {
        statusBarItem.dispose();
    }
    if (outputChannel) {
        outputChannel.dispose();
    }
    if (debounceTimer) {
        clearTimeout(debounceTimer);
    }
}
