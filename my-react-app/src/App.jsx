import { useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';
import AutocompleteInput from './AutomatedInput.jsx';
import Navbar from './Navbar.jsx';

function App() {
  return (
    <div className="app-container">
      <Navbar />
      <AutocompleteInput />
    </div>
  );
}

export default App;
