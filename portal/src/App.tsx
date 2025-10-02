import React, { useState } from 'react';
import Header from './components/Header';
import HomePage from './components/HomePage';
import AIChat from './components/AIChat';
import { IaChatProvider } from './context/iachatContext';
import ContextStatus from './components/ContextStatus';

function App() {
  const [activeSection, setActiveSection] = useState('home');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const handleSectionChange = (section: string) => {
    setActiveSection(section);
    setIsMobileMenuOpen(false);
  };

  const handleMobileMenuToggle = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'home':
        return <HomePage onSectionChange={handleSectionChange} />;
      case 'chat':
        return <AIChat />;
      default:
        return <HomePage onSectionChange={handleSectionChange} />;
    }
  };

  return (
    <IaChatProvider>
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50">
      <Header 
        activeSection={activeSection}
        onSectionChange={handleSectionChange}
        isMobileMenuOpen={isMobileMenuOpen}
        onMobileMenuToggle={handleMobileMenuToggle}
      />
      <main className="pb-8">
       {/*  <ContextStatus /> */}
        {renderActiveSection()}
      </main>
    </div>
    </IaChatProvider>
  );
}

export default App;