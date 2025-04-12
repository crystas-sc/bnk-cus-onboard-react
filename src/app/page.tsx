"use client"
import Image from "next/image";
import styles from "./page.module.css";
import React, { useState, useCallback, useRef, useEffect } from 'react';

import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import {
    MessageSquare,
    Send,
    Loader2,
    CheckCircle,
    XCircle,
    Banknote,
    Briefcase,
    FileText,
    Users,
    Settings,
    LogOut,
    Menu,
    X,
} from 'lucide-react';
import { cn } from '@/lib/utils';


export default function Home() {
  return (
   <CorporateBankOnboardingApp />

        );
}



// Mock API for AI Chat (replace with actual API calls)
const mockAIChatAPI = {
  sendMessage: async (message: string, route?: string) => {
      await new Promise((resolve) => setTimeout(resolve, 500)); // Simulate network delay

      if (route === 'kyc') {
          return {
              text: 'For KYC, please provide your business registration number and address.',
              route: 'kyc',
          };
      } else if (route === 'account_setup') {
          return {
              text: 'To set up your corporate account, we need your company name and contact details.',
              route: 'account_setup',
          };
      } else if (route === 'loan_application') {
          return {
              text: "For loan applications, please provide your business's financial statements.",
              route: 'loan_application'
          }
      } else {
          return {
              text: `Thank you for your message: "${message}". How can I assist you further?  You can ask me about KYC, account setup, or loan applications.  You can also use the buttons below.`,
          };
      }
  },
};

// Animation Variants
const messageVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.2 } },
  exit: { opacity: 0, y: -10, transition: { duration: 0.1 } },
};

// Chat Message Component
const ChatMessage = ({ message }: { message: { text: string; sender: 'user' | 'ai'; route?: string } }) => {
  const getMessageIcon = (sender: 'user' | 'ai') => {
      if (sender === 'ai') {
          return <Banknote className="w-4 h-4 mr-2 text-blue-500" />;
      }
      return <Users className="w-4 h-4 mr-2 text-green-500" />;
  };

  return (
      <motion.div
          variants={messageVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
          className={cn(
              'flex items-start gap-3 mb-4',
              message.sender === 'user' ? 'justify-end' : 'justify-start'
          )}
      >
          {message.sender === 'ai' && getMessageIcon('ai')}
          <div
              className={cn(
                  'rounded-lg px-4 py-3 max-w-[70%]',
                  message.sender === 'user'
                      ? 'bg-green-500/20 text-green-200 ml-auto'
                      : 'bg-blue-500/20 text-blue-200',
                  'shadow-md'
              )}
          >
              <p className="text-sm whitespace-pre-wrap break-words">{message.text}</p>
          </div>
          {message.sender === 'user' && getMessageIcon('user')}
      </motion.div>
  );
};

// Chat Widget Component
const ChatWidget = ({ onRoute }: { onRoute?: (route: string) => void }) => {
  const [messages, setMessages] = useState<{ text: string; sender: 'user' | 'ai'; route?: string }[]>([]);
  const [input, setInput] = useState('');
  const [isAITyping, setIsAITyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
//   const suggestionButtons = [
//       { label: 'KYC', route: 'kyc' },
//       { label: 'Account Setup', route: 'account_setup' },
//       { label: 'Loan Application', route: 'loan_application' },
//   ];
  const [suggestionButtons, setSuggestionButtons] = useState([]);

  const socketRef = useRef(null) // 👈 This line declares the WebSocket ref
  const [msgJson, setMsgJson] = useState({
    user_id: "rob42",
    thread_id: "alpha-42",
    input: ""
  }) // 👈 This line declares the message JSON state


  // Scroll to bottom on new message
  useEffect(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.host
    socketRef.current = new WebSocket(`${protocol}://${host}/socket`)
  
    socketRef.current.onopen = () => {
      console.log('WebSocket connected')
      const message = structuredClone(msgJson)
      message.input = "Hi"
      socketRef.current.send(JSON.stringify(message))
    }
  
    socketRef.current.onmessage = (event) => {
      console.log('Message from server:', event.data)
      const msg = JSON.parse(event.data)
    //   setMessages(prev => [...prev, event.data])
        setMessages((prevMessages) => [...prevMessages, {text:msg.message, sender:"ai"}]);
        setSuggestionButtons(msg.suggestions);
        setIsAITyping(false);
    
    }
  
    socketRef.current.onclose = () => {
      console.log('WebSocket disconnected')
    }
  
    socketRef.current.onerror = (err) => {
      console.error('WebSocket error:', err)
    }
  
    return () => {
      socketRef.current.close()
    }
  }, [])

  const handleSendMessage = useCallback(async () => {
      if (!input.trim()) return;
    

      const userMessage = { text: input, sender: 'user' };
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setInput('');
      setIsAITyping(true);
      const message = structuredClone(msgJson)
      message.input = input
      socketRef.current.send(JSON.stringify(message))
      return;

      try {
        //   const aiResponse = await mockAIChatAPI.sendMessage(input);
            let aiResponse = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({"user_id": "rob42", "thread_id": "alpha-42", "input": input}),
            }).then((res) => res.text());
            console.log("AI response", aiResponse);
            aiResponse = JSON.parse(aiResponse);
            
          setMessages((prevMessages) => [...prevMessages, {text:aiResponse.output, sender:"ai"}]);
          if (aiResponse.route && onRoute) {
              onRoute(aiResponse.route);
          }
      } catch (error) {
          console.error('Error sending message:', error);
          const errorMessage = {
              text: 'Sorry, there was an error processing your request.',
              sender: 'ai',
          };
          setMessages((prevMessages) => [...prevMessages, errorMessage]);
      } finally {
          setIsAITyping(false);
      }
  }, [input, onRoute]);

  const handleSuggestionClick = (route: string) => {
      onRoute?.(route);
      setIsChatOpen(false);
  };

  return (
      <div className="flex flex-col h-full">
          {/* Chat Messages Area */}
          <div className="flex-1 overflow-y-auto p-4">
              <AnimatePresence>
                  {messages.map((message, index) => (
                      <ChatMessage key={index} message={message} />
                  ))}
              </AnimatePresence>
              {isAITyping && (
                  <motion.div
                      variants={messageVariants}
                      initial="hidden"
                      animate="visible"
                      className="flex items-center gap-3 mb-4"
                  >
                      <Banknote className="w-4 h-4 mr-2 text-blue-500" />
                      <div className="bg-blue-500/20 text-blue-200 rounded-lg px-4 py-3 max-w-[70%] shadow-md">
                          <Loader2 className="animate-spin w-5 h-5" />
                      </div>
                  </motion.div>
              )}
              <div ref={messagesEndRef} />
          </div>

          {/* Suggestion Buttons */}
          <div className="p-4 space-x-2 flex flex-wrap justify-center">
              {suggestionButtons.map((button) => (
                  <Button
                      key={button.route}
                      onClick={() => handleSuggestionClick(button.route)}
                      className="bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 hover:text-purple-200 rounded-full px-4 py-2 transition-colors"
                  >
                      {button.label}
                  </Button>
              ))}
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-700">
              <div className="flex gap-2">
                  <Textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                              e.preventDefault();
                              handleSendMessage();
                          }
                      }}
                      placeholder="Type your message..."
                      className="flex-1 bg-gray-800 text-white border-gray-600 rounded-md p-3 resize-none min-h-[2.5rem] focus:outline-none focus:ring-2 focus:ring-blue-500"
                      rows={1} // Start with one row
                      style={{ maxHeight: '10rem' }} // Limit the maximum height
                  />
                  <Button
                      onClick={handleSendMessage}
                      className="bg-blue-500 text-white rounded-md px-6 py-3 hover:bg-blue-600 transition-colors"
                      disabled={isAITyping}
                  >
                      {isAITyping ? (
                          <Loader2 className="animate-spin w-5 h-5" />
                      ) : (
                          <Send className="w-5 h-5" />
                      )}
                  </Button>
              </div>
          </div>
      </div>
  );
};

// KYC Form Component
const KYCForm = () => {
  const [formData, setFormData] = useState({
      businessName: '',
      registrationNumber: '',
      address: '',
      contactPerson: '',
      email: '',
      phone: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStatus, setSubmissionStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();
      setIsSubmitting(true);
      setSubmissionStatus('idle');

      // Simulate API call
      try {
          await new Promise((resolve) => setTimeout(resolve, 2000));
          // Replace with actual API call
          console.log('KYC Form Data:', formData);
          setSubmissionStatus('success');
          setFormData({
              businessName: '',
              registrationNumber: '',
              address: '',
              contactPerson: '',
              email: '',
              phone: '',
          }); // Clear form on success
      } catch (error) {
          console.error('KYC Form Submission Error:', error);
          setSubmissionStatus('error');
      } finally {
          setIsSubmitting(false);
      }
  };

  return (
      <div className="bg-gray-900 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-white mb-6">Know Your Customer (KYC)</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                  <Label htmlFor="businessName" className="text-gray-300">
                      Business Name
                  </Label>
                  <Input
                      type="text"
                      id="businessName"
                      name="businessName"
                      value={formData.businessName}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter your business name"
                  />
              </div>
              <div>
                  <Label htmlFor="registrationNumber" className="text-gray-300">
                      Registration Number
                  </Label>
                  <Input
                      type="text"
                      id="registrationNumber"
                      name="registrationNumber"
                      value={formData.registrationNumber}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter registration number"
                  />
              </div>
              <div>
                  <Label htmlFor="address" className="text-gray-300">
                      Address
                  </Label>
                  <Textarea
                      id="address"
                      name="address"
                      value={formData.address}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter business address"
                  />
              </div>
              <div>
                  <Label htmlFor="contactPerson" className="text-gray-300">
                      Contact Person
                  </Label>
                  <Input
                      type="text"
                      id="contactPerson"
                      name="contactPerson"
                      value={formData.contactPerson}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter contact person name"
                  />
              </div>
              <div>
                  <Label htmlFor="email" className="text-gray-300">
                      Email
                  </Label>
                  <Input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter email address"
                  />
              </div>
              <div>
                  <Label htmlFor="phone" className="text-gray-300">
                      Phone Number
                  </Label>
                  <Input
                      type="tel"
                      id="phone"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter phone number"
                  />
              </div>
              <Button
                  type="submit"
                  className="w-full bg-blue-500 text-white py-3 rounded-md hover:bg-blue-600 transition-colors"
                  disabled={isSubmitting}
              >
                  {isSubmitting ? (
                      <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Submitting...
                      </>
                  ) : (
                      'Submit KYC Information'
                  )}
              </Button>
              {submissionStatus === 'success' && (
                  <div className="bg-green-500/20 text-green-400 p-3 rounded-md flex items-center">
                      <CheckCircle className="mr-2 w-5 h-5" />
                      KYC information submitted successfully!
                  </div>
              )}
              {submissionStatus === 'error' && (
                  <div className="bg-red-500/20 text-red-400 p-3 rounded-md flex items-center">
                      <XCircle className="mr-2 w-5 h-5" />
                      Error submitting KYC information. Please try again.
                  </div>
              )}
          </form>
      </div>
  );
};

// Account Setup Form Component
const AccountSetupForm = () => {
  const [formData, setFormData] = useState({
      companyName: '',
      companyAddress: '',
      contactName: '',
      contactEmail: '',
      contactPhone: '',
      accountType: 'checking', // Default value
      initialDeposit: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStatus, setSubmissionStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
      setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();
      setIsSubmitting(true);
      setSubmissionStatus('idle');

      // Simulate API call
      try {
          await new Promise((resolve) => setTimeout(resolve, 2000));
          // Replace with actual API call
          console.log('Account Setup Form Data:', formData);
          setSubmissionStatus('success');
          setFormData({
              companyName: '',
              companyAddress: '',
              contactName: '',
              contactEmail: '',
              contactPhone: '',
              accountType: 'checking',
              initialDeposit: '',
          }); // Clear form on success
      } catch (error) {
          console.error('Account Setup Form Submission Error:', error);
          setSubmissionStatus('error');
      } finally {
          setIsSubmitting(false);
      }
  };

  return (
      <div className="bg-gray-900 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-white mb-6">Set Up Your Corporate Account</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                  <Label htmlFor="companyName" className="text-gray-300">
                      Company Name
                  </Label>
                  <Input
                      type="text"
                      id="companyName"
                      name="companyName"
                      value={formData.companyName}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter company name"
                  />
              </div>
              <div>
                  <Label htmlFor="companyAddress" className="text-gray-300">
                      Company Address
                  </Label>
                  <Textarea
                      id="companyAddress"
                      name="companyAddress"
                      value={formData.companyAddress}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter company address"
                  />
              </div>
              <div>
                  <Label htmlFor="contactName" className="text-gray-300">
                      Contact Name
                  </Label>
                  <Input
                      type="text"
                      id="contactName"
                      name="contactName"
                      value={formData.contactName}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter contact person name"
                  />
              </div>
              <div>
                  <Label htmlFor="contactEmail" className="text-gray-300">
                      Contact Email
                  </Label>
                  <Input
                      type="email"
                      id="contactEmail"
                      name="contactEmail"
                      value={formData.contactEmail}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter contact email"
                  />
              </div>
              <div>
                  <Label htmlFor="contactPhone" className="text-gray-300">
                      Contact Phone
                  </Label>
                  <Input
                      type="tel"
                      id="contactPhone"
                      name="contactPhone"
                      value={formData.contactPhone}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter contact phone number"
                  />
              </div>
              <div>
                  <Label htmlFor="accountType" className="text-gray-300">
                      Account Type
                  </Label>
                  <select
                      id="accountType"
                      name="accountType"
                      value={formData.accountType}
                      onChange={handleChange}
                      className="bg-gray-800 text-white border-gray-700 rounded-md p-3 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                      <option value="checking">Checking Account</option>
                      <option value="savings">Savings Account</option>
                      <option value="moneyMarket">Money Market Account</option>
                  </select>
              </div>
              <div>
                  <Label htmlFor="initialDeposit" className="text-gray-300">
                      Initial Deposit
                  </Label>
                  <Input
                      type="number"
                      id="initialDeposit"
                      name="initialDeposit"
                      value={formData.initialDeposit}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter initial deposit amount"
                  />
              </div>
              <Button
                  type="submit"
                  className="w-full bg-blue-500 text-white py-3 rounded-md hover:bg-blue-600 transition-colors"
                  disabled={isSubmitting}
              >
                  {isSubmitting ? (
                      <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Submitting...
                      </>
                  ) : (
                      'Submit Account Setup'
                  )}
              </Button>
              {submissionStatus === 'success' && (
                  <div className="bg-green-500/20 text-green-400 p-3 rounded-md flex items-center">
                      <CheckCircle className="mr-2 w-5 h-5" />
                      Account setup information submitted successfully!
                  </div>
              )}
              {submissionStatus === 'error' && (
                  <div className="bg-red-500/20 text-red-400 p-3 rounded-md flex items-center">
                      <XCircle className="mr-2 w-5 h-5" />
                      Error submitting account setup information. Please try again.
                  </div>
              )}
          </form>
      </div>
  );
};

const LoanApplicationForm = () => {
  const [formData, setFormData] = useState({
      businessName: '',
      loanAmount: '',
      loanPurpose: '',
      financialStatements: '', // Could be a file input in a real scenario
      collateral: '',
      revenue: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStatus, setSubmissionStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setFormData({ ...formData, [e.target.name]: e.target.value });
  };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
          // In a real application, you would handle file uploads here (e.g., to cloud storage)
          const file = e.target.files[0];
          setFormData({ ...formData, financialStatements: file.name }); // Store file name for display
          console.log("Uploaded file", file);
      }
  };

  const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();
      setIsSubmitting(true);
      setSubmissionStatus('idle');

      // Simulate API call
      try {
          await new Promise((resolve) => setTimeout(resolve, 2000));
          // Replace with actual API call
          console.log('Loan Application Form Data:', formData);
          setSubmissionStatus('success');
          setFormData({
              businessName: '',
              loanAmount: '',
              loanPurpose: '',
              financialStatements: '',
              collateral: '',
              revenue: '',
          }); // Clear form on success
      } catch (error) {
          console.error('Loan Application Form Submission Error:', error);
          setSubmissionStatus('error');
      } finally {
          setIsSubmitting(false);
      }
  };

  return (
      <div className="bg-gray-900 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-white mb-6">Apply for a Loan</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                  <Label htmlFor="businessName" className="text-gray-300">
                      Business Name
                  </Label>
                  <Input
                      type="text"
                      id="businessName"
                      name="businessName"
                      value={formData.businessName}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter business name"
                  />
              </div>
              <div>
                  <Label htmlFor="loanAmount" className="text-gray-300">
                      Loan Amount
                  </Label>
                  <Input
                      type="number"
                      id="loanAmount"
                      name="loanAmount"
                      value={formData.loanAmount}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter loan amount"
                  />
              </div>
              <div>
                  <Label htmlFor="loanPurpose" className="text-gray-300">
                      Loan Purpose
                  </Label>
                  <Textarea
                      id="loanPurpose"
                      name="loanPurpose"
                      value={formData.loanPurpose}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter loan purpose"
                  />
              </div>
              <div>
                <Label htmlFor="financialStatements" className="text-gray-300">
                  Financial Statements
                </Label>
                <Input
                  type="file"
                  id="financialStatements"
                  name="financialStatements"
                  onChange={handleFileChange}
                  required
                  className="bg-gray-800 text-white border-gray-700"
                  placeholder="Upload financial statements"
                />
                {formData.financialStatements && (
                  <p className="text-sm text-gray-400">
                    Uploaded: {formData.financialStatements}
                  </p>
                )}
              </div>
              <div>
                  <Label htmlFor="collateral" className="text-gray-300">
                      Collateral
                  </Label>
                  <Input
                      type="text"
                      id="collateral"
                      name="collateral"
                      value={formData.collateral}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter collateral details"
                  />
              </div>
              <div>
                  <Label htmlFor="revenue" className="text-gray-300">
                      Annual Revenue
                  </Label>
                  <Input
                      type="number"
                      id="revenue"
                      name="revenue"
                      value={formData.revenue}
                      onChange={handleChange}
                      required
                      className="bg-gray-800 text-white border-gray-700"
                      placeholder="Enter annual revenue"
                  />
              </div>
              <Button
                  type="submit"
                  className="w-full bg-blue-500 text-white py-3 rounded-md hover:bg-blue-600 transition-colors"
                  disabled={isSubmitting}
              >
                  {isSubmitting ? (
                      <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Submitting...
                      </>
                  ) : (
                      'Submit Loan Application'
                  )}
              </Button>
              {submissionStatus === 'success' &&(
                  <div className="bg-green-500/20 text-green-400 p-3 rounded-md flex items-center">
                      <CheckCircle className="mr-2 w-5 h-5" />
                      Loan application submitted successfully!
                  </div>
              )}
              {submissionStatus === 'error' && (
                  <div className="bg-red-500/20 text-red-400 p-3 rounded-md flex items-center">
                      <XCircle className="mr-2 w-5 h-5" />
                      Error submitting loan application. Please try again.
                  </div>
              )}
          </form>
      </div>
  );
};

// Main App Component
const CorporateBankOnboardingApp = () => {
  const [activeSection, setActiveSection] = useState<'home' | 'kyc' | 'account_setup' | 'loan_application' | 'chat'>('home');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false); // State for mobile menu

  const handleChatRoute = (route: string) => {
      if (route === 'kyc') {
          setActiveSection('kyc');
          setIsChatOpen(false); // Close chat after routing
      } else if (route === 'account_setup') {
          setActiveSection('account_setup');
          setIsChatOpen(false);
      } else if (route === 'loan_application') {
          setActiveSection('loan_application');
          setIsChatOpen(false);
      } else {
          setActiveSection('home');
          setIsChatOpen(false);
      }
  };

  const toggleMobileMenu = () => {
      setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
      <div className="min-h-screen bg-gray-950 text-white flex flex-col">
          {/* Header */}
          <header className="bg-gray-900 border-b border-gray-800 py-4 px-6 flex items-center justify-between">
              <div className="flex items-center">
                  <Banknote className="w-8 h-8 mr-2 text-blue-500" />
                  <h1 className="text-2xl font-bold">Corporate Bank Onboarding</h1>
              </div>

              {/* Mobile Menu Button */}
              <div className="md:hidden">
                  <Button
                      variant="ghost"
                      className="text-gray-300 hover:text-white"
                      onClick={toggleMobileMenu}
                      aria-label="Toggle Mobile Menu"
                  >
                      {isMobileMenuOpen ? (
                          <X className="w-6 h-6" />
                      ) : (
                          <Menu className="w-6 h-6" />
                      )}
                  </Button>
              </div>

              {/* Navigation (Desktop) */}
              <nav className="hidden md:block">
                  <ul className="flex space-x-6">
                      <li>
                          <Button
                              variant={activeSection === 'home' ? 'default' : 'ghost'}
                              className={cn(
                                  "text-gray-300 hover:text-white",
                                  activeSection === 'home' && "bg-blue-500 text-white"
                              )}
                              onClick={() => setActiveSection('home')}
                          >
                              Home
                          </Button>
                      </li>
                      <li>
                          <Button
                              variant={activeSection === 'kyc' ? 'default' : 'ghost'}
                              className={cn(
                                  "text-gray-300 hover:text-white",
                                  activeSection === 'kyc' && "bg-blue-500 text-white"
                              )}
                              onClick={() => setActiveSection('kyc')}
                          >
                              KYC
                          </Button>
                      </li>
                      <li>
                          <Button
                              variant={activeSection === 'account_setup' ? 'default' : 'ghost'}
                              className={cn(
                                  "text-gray-300 hover:text-white",
                                  activeSection === 'account_setup' && "bg-blue-500 text-white"
                              )}
                              onClick={() => setActiveSection('account_setup')}
                          >
                              Account Setup
                          </Button>
                      </li>
                      <li>
                          <Button
                              variant={activeSection === 'loan_application' ? 'default' : 'ghost'}
                              className={cn(
                                "text-gray-300 hover:text-white",
                                activeSection === 'loan_application' && "bg-blue-500 text-white"
                              )}
                              onClick={() => setActiveSection('loan_application')}
                          >
                              Loan Application
                          </Button>
                      </li>
                  </ul>
              </nav>
          </header>

          {/* Mobile Menu */}
          <AnimatePresence>
              {isMobileMenuOpen && (
                  <motion.nav
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="md:hidden bg-gray-900 border-b border-gray-800 py-4"
                  >
                      <ul className="space-y-4 px-6">
                          <li>
                              <Button
                                  variant={activeSection === 'home' ? 'default' : 'ghost'}
                                  className={cn(
                                      "text-gray-300 hover:text-white w-full justify-start",
                                      activeSection === 'home' && "bg-blue-500 text-white"
                                  )}
                                  onClick={() => {
                                      setActiveSection('home');
                                      setIsMobileMenuOpen(false);
                                  }}
                              >
                                  Home
                              </Button>
                          </li>
                          <li>
                              <Button
                                  variant={activeSection === 'kyc' ? 'default' : 'ghost'}
                                  className={cn(
                                      "text-gray-300 hover:text-white w-full justify-start",
                                      activeSection === 'kyc' && "bg-blue-500 text-white"
                                  )}
                                  onClick={() => {
                                      setActiveSection('kyc');
                                      setIsMobileMenuOpen(false);
                                  }}
                              >
                                  KYC
                              </Button>
                          </li>
                          <li>
                              <Button
                                  variant={activeSection === 'account_setup' ? 'default' : 'ghost'}
                                  className={cn(
                                      "text-gray-300 hover:text-white w-full justify-start",
                                      activeSection === 'account_setup' && "bg-blue-500 text-white"
                                  )}
                                  onClick={() => {
                                      setActiveSection('account_setup');
                                      setIsMobileMenuOpen(false);
                                  }}
                              >
                                  Account Setup
                              </Button>
                          </li>
                          <li>
                              <Button
                                  variant={activeSection === 'loan_application' ? 'default' : 'ghost'}
                                  className={cn(
                                    "text-gray-300 hover:text-white w-full justify-start",
                                    activeSection === 'loan_application' && "bg-blue-500 text-white"
                                  )}
                                  onClick={() => {
                                      setActiveSection('loan_application');
                                      setIsMobileMenuOpen(false);
                                  }}
                              >
                                  Loan Application
                              </Button>
                          </li>
                      </ul>
                  </motion.nav>
              )}
          </AnimatePresence>

          {/* Main Content */}
          <main className="flex-1 p-6">
              <AnimatePresence mode="wait">
                  {activeSection === 'home' && (
                      <motion.div
                          key="home"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          className="text-center"
                      >
                          <h2 className="text-3xl font-semibold mb-4">Welcome to Corporate Bank Onboarding</h2>
                          <p className="text-gray-300 mb-8">
                              Streamline your onboarding process with our digital solutions.
                          </p>
                          <Button
                              onClick={() => setActiveSection('kyc')}
                              className="bg-blue-500 text-white px-8 py-3 rounded-md hover:bg-blue-600 transition-colors mr-4"
                          >
                              Complete KYC
                          </Button>
                          <Button
                              onClick={() => setActiveSection('account_setup')}
                              className="bg-green-500 text-white px-8 py-3 rounded-md hover:bg-green-600 transition-colors"
                          >
                              Set Up Account
                          </Button>
                          <Button
                              onClick={() => setActiveSection('loan_application')}
                              className="bg-purple-500 text-white px-8 py-3 rounded-md hover:bg-purple-600 transition-colors ml-4"
                          >
                              Apply for Loan
                          </Button>
                      </motion.div>
                  )}
                  {activeSection === 'kyc' && (
                      <motion.div
                          key="kyc"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                      >
                          <KYCForm />
                      </motion.div>
                  )}
                  {activeSection === 'account_setup' && (
                      <motion.div
                          key="account_setup"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                      >
                          <AccountSetupForm />
                      </motion.div>
                  )}
                  {activeSection === 'loan_application' && (
                      <motion.div
                          key="loan_application"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                      >
                          <LoanApplicationForm />
                      </motion.div>
                  )}
              </AnimatePresence>
          </main>

          {/* Footer */}
          <footer className="bg-gray-900 border-t border-gray-800 py-4 px-6 text-center text-gray-400">
              &copy; {new Date().getFullYear()} Corporate Bank. All rights reserved.
          </footer>

          {/* Chat Widget Toggle */}
          <div className="fixed bottom-4 right-4 z-50">
              <Button
                  onClick={() => setIsChatOpen(!isChatOpen)}
                  className={cn(
                      'rounded-full p-3 shadow-lg',
                      isChatOpen ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'
                  )}
                  aria-label="Toggle Chat Widget"
              >
                  {isChatOpen ? (
                      <X className="w-6 h-6 text-white" />
                  ) : (
                      <MessageSquare className="w-6 h-6 text-white" />
                  )}
              </Button>
          </div>

          {/* Chat Widget */}
          <AnimatePresence>
              {isChatOpen && (
                  <motion.div
                      initial={{ opacity: 0, y: 50 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 50 }}
                      className="fixed bottom-20 right-4 z-50 w-full max-w-md h-[400px] bg-gray-900 rounded-lg shadow-2xl border border-gray-800"
                  >
                      <ChatWidget onRoute={handleChatRoute} />
                  </motion.div>
              )}
          </AnimatePresence>
      </div>
  );
};
