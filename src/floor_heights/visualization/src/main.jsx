import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import Annotator from './Annotator'
import './index.css'

const urlParams = new URLSearchParams(window.location.search);
const mode = urlParams.get('mode');

const AppToRender = mode === 'annotate' ? Annotator : App;

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppToRender />
  </React.StrictMode>,
)
