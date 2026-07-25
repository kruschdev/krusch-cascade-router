import { spawn } from 'child_process';
import path from 'path';

// This prototype demonstrates using the heuristic classification pattern from
// krusch-cascade-router to orchestrate Macro (pgvector) vs Micro (PageIndex) RAG paths.

const customRules = [
  // Rules to detect if the user is asking about a specific document deep-dive
  (prompt) => /in document/i.test(prompt),
  (prompt) => /on page/i.test(prompt),
  (prompt) => /specific to the/i.test(prompt),
  (prompt) => /what does the .* policy say/i.test(prompt),
  (prompt) => /handbook/i.test(prompt),
];

function isMicroIntent(prompt) {
  return customRules.some(rule => rule(prompt));
}

async function runPageIndexWorker(prompt) {
  return new Promise((resolve, reject) => {
    const workerPath = path.join(process.cwd(), 'scratch', 'pageindex_worker.py');
    // Using the local venv we created
    const pythonExec = path.join(process.cwd(), 'scratch', 'venv', 'bin', 'python3');
    
    const child = spawn(pythonExec, [workerPath, prompt]);
    
    let output = '';
    let errorOutput = '';

    child.stdout.on('data', (data) => {
      output += data.toString();
      // Stream output directly to console for the agentic streaming effect
      process.stdout.write(data.toString());
    });

    child.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`PageIndex worker failed with code ${code}: ${errorOutput}`));
      }
    });
  });
}

async function mockPgvectorSearch(prompt) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const result = `\n[pgvector]: Searched 10,000 documents in 45ms.\n[pgvector]: The most relevant document for your query is "2025_Employee_Handbook.pdf" (Document ID: DOC-992).`;
      console.log(result);
      resolve(result);
    }, 1000);
  });
}

async function main() {
  const query = process.argv[2] || "What does the 2025 Employee Handbook say about PTO?";
  
  console.log(`==================================================`);
  console.log(`📝 Query: "${query}"`);
  console.log(`==================================================`);
  
  if (isMicroIntent(query)) {
    console.log(`[CascadeRouter] ⚡ Heuristics classified intent as MICRO (Deep Document Search).`);
    console.log(`[CascadeRouter] 🔀 Routing to PageIndex Python Worker...`);
    await runPageIndexWorker(query);
  } else {
    console.log(`[CascadeRouter] ⚡ Heuristics classified intent as MACRO (Broad Corpus Search).`);
    console.log(`[CascadeRouter] 🔀 Routing to pgvector fast-path...`);
    await mockPgvectorSearch(query);
  }
}

main().catch(console.error);
