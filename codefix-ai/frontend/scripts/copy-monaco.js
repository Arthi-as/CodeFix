// Copy the installed monaco-editor runtime into public/vs.
//
// @monaco-editor/react defaults to loading Monaco from a CDN, which fails
// without internet access and blocks the editor from ever mounting. Serving the
// files from this origin instead makes the app work offline.

const fs = require('fs');
const path = require('path');

const source = path.join(__dirname, '..', 'node_modules', 'monaco-editor', 'min', 'vs');
const target = path.join(__dirname, '..', 'public', 'vs');

if (!fs.existsSync(source)) {
  console.error(`monaco-editor not found at ${source}`);
  console.error('Run `npm install` first.');
  process.exit(1);
}

fs.rmSync(target, { recursive: true, force: true });
fs.mkdirSync(path.dirname(target), { recursive: true });
fs.cpSync(source, target, { recursive: true });

const files = fs.readdirSync(target).length;
console.log(`copied monaco-editor -> public/vs (${files} top-level entries)`);