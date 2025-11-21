// VSCode Extension - Sigma-C
// ===========================
// Copyright (c) 2025 ForgottenForge.xyz

import * as vscode from 'vscode';
import { exec } from 'child_process';

let statusBarItem: vscode.StatusBarItem;
let isMonitoring = false;

export function activate(context: vscode.ExtensionContext) {
    console.log('Sigma-C extension activated');
    
    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = 'σ_c: --';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
    
    // Register commands
    let analyzeCommand = vscode.commands.registerCommand(
        'sigma-c.analyze',
        analyzeCurrentFile
    );
    
    let toggleCommand = vscode.commands.registerCommand(
        'sigma-c.toggleMonitoring',
        toggleMonitoring
    );
    
    context.subscriptions.push(analyzeCommand, toggleCommand);
    
    // Auto-analyze on save
    vscode.workspace.onDidSaveTextDocument(document => {
        if (isMonitoring) {
            analyzeDocument(document);
        }
    });
}

async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return;
    }
    
    await analyzeDocument(editor.document);
}

async function analyzeDocument(document: vscode.TextDocument) {
    const text = document.getText();
    
    // Compute simple complexity metrics
    const lines = text.split('\n');
    const nonEmptyLines = lines.filter(l => l.trim().length > 0).length;
    const avgLineLength = lines.reduce((sum, l) => sum + l.length, 0) / lines.length;
    
    // Simple criticality metric
    const sigma_c = Math.min(1.0, (avgLineLength / 80) * (nonEmptyLines / 100));
    
    // Update status bar
    statusBarItem.text = `σ_c: ${sigma_c.toFixed(3)}`;
    
    // Show warning if high
    const threshold = vscode.workspace.getConfiguration('sigma-c').get('threshold', 0.7);
    if (sigma_c > threshold) {
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        vscode.window.showWarningMessage(
            `High code criticality detected: σ_c = ${sigma_c.toFixed(3)}`
        );
    } else {
        statusBarItem.backgroundColor = undefined;
    }
}

function toggleMonitoring() {
    isMonitoring = !isMonitoring;
    vscode.window.showInformationMessage(
        `Sigma-C monitoring ${isMonitoring ? 'enabled' : 'disabled'}`
    );
}

export function deactivate() {
    if (statusBarItem) {
        statusBarItem.dispose();
    }
}
