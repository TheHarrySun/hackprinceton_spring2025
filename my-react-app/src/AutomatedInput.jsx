import { useState, useEffect } from 'react';
import Papa from 'papaparse'; // Import PapaParse library
import './Autocomplete.css'; // Import the CSS file

const AutocompleteInput = () => {
  const [input, setInput] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isValid, setIsValid] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [dictionary, setDictionary] = useState({});
  const [submittedOptions, setSubmittedOptions] = useState([]);
  const [maxLimitReached, setMaxLimitReached] = useState(false);
  const [drugInfo, setDrugInfo] = useState(null);
  const [loadingInfo, setLoadingInfo] = useState(false);
  const [loadingInteractions, setLoadingInteractions] = useState(false);

  const [interactionData, setInteractionData] = useState(null);

  useEffect(() => {
    Papa.parse('/cid_synonyms.csv', {
      download: true,
      complete: (result) => {
        const newDictionary = {};
        result.data.forEach((row) => {
          const key = row[0];
          const value = row[1];
          if (key && value) {
            newDictionary[key] = value;
          }
        });
        setDictionary(newDictionary);
      },
    });
  }, []);

  const handleChange = (e) => {
    const value = e.target.value;
    setInput(value);

    if (value) {
      const filteredSuggestions = Object.keys(dictionary)
        .filter((key) =>
          key.toLowerCase().startsWith(value.toLowerCase())
        )
        .slice(0, 5);
      setSuggestions(filteredSuggestions);
      setIsValid(filteredSuggestions.includes(value));
    } else {
      setSuggestions([]);
      setIsValid(false);
    }
  };

  const handleSelect = (item) => {
    setInput(item);
    setSuggestions([]);
    setIsValid(true);
    setSelectedOption(item);
  };

  const fetchDrugInfo = async (drug) => {
    setLoadingInfo(true);
    try {
      const response = await fetch('http://localhost:4000/api/query-gemini-info', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ drugName: drug }),
      });

      const data = await response.json();
      setDrugInfo(data.data || 'No information available for this drug.');
    } catch (error) {
      setDrugInfo('Failed to fetch drug information.');
    } finally {
      setLoadingInfo(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isValid && !maxLimitReached) {
      if (submittedOptions.includes(selectedOption)) {
        alert('This option has already been submitted!');
        return;
      }
      setSubmittedOptions((prev) => {
        const updatedOptions = [...prev, selectedOption];
        if (updatedOptions.length >= 5) {
          setMaxLimitReached(true);
        }
        return updatedOptions;
      });
      setInput('');
      setSuggestions([]);
      setIsValid(false);
    } else if (maxLimitReached) {
      alert('You can\'t add more than 5 options!');
    } else {
      alert('Please select a valid option!');
    }
  };

  const handleRemove = (index) => {
    setSubmittedOptions((prev) => {
      const updated = [...prev];
      updated.splice(index, 1);
      return updated;
    });
    setMaxLimitReached(false);
  };

  const handleClickOnSubmittedItem = (item) => {
    fetchDrugInfo(item);
  };


  const handleFetchInteractions = async () => {
    if (submittedOptions.length === 0) {
      alert('Please add drugs to the list first!');
      return;
    }

    setLoadingInteractions(true);
    try {
      const response = await fetch('http://localhost:4000/api/query-gemini-interactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ drugList: submittedOptions }),
      });

      const data = await response.json();
      if (data.data) {
        // Stringify the JSON data with formatting
        const jsonString = JSON.stringify(data.data, null, 2);
        
        // Replace \n with actual line breaks
        const formattedData = jsonString.replace(/\\n/g, '<br />');

        const cleanedData = formattedData.slice(1, -1);  // Removes the first and last character

        
        // Set the formatted data with HTML line breaks
        setInteractionData(cleanedData);
        
        console.log(formattedData);
      } else {
        setInteractionData('No interaction data available.');
      }
    } catch (error) {
      console.error('Error fetching interaction data:', error);
      setInteractionData('Failed to fetch interaction data.');
    } finally {
      setLoadingInteractions(false);
    }
  };

  return (
    <div className="main-container">
      {/* Left Side: Search and Submitted Options */}
      <div className="left-container">
        <div className="autocomplete-container">
          <form onSubmit={handleSubmit} className="autocomplete-form">
            <div className="input-button-wrapper">
              <input
                type="text"
                value={input}
                onChange={handleChange}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                className="autocomplete-input"
                placeholder="Type to search..."
              />
              <button
                type="submit"
                className="submit-button"
                disabled={!isValid || maxLimitReached}
              >
                Submit
              </button>
            </div>
            {isFocused && suggestions.length > 0 && (
              <ul className="autocomplete-dropdown">
                {suggestions.map((item, index) => (
                  <li
                    key={index}
                    className="autocomplete-item"
                    onMouseDown={() => handleSelect(item)}
                  >
                    {item}
                  </li>
                ))}
              </ul>
            )}
          </form>
        </div>
  
        {/* Submitted Options */}
        <div className="submitted-options">
          <h3>Submitted Options:</h3>
          <ul>
            {submittedOptions.map((option, index) => (
              <li
                key={index}
                className="submitted-item"
                onClick={() => handleClickOnSubmittedItem(option)}
              >
                <div className="submitted-item-box">{option}</div>
                <button
                  type="button"
                  className="remove-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemove(index);
                  }}
                >
                  &times;
                </button>
              </li>
            ))}
          </ul>
          {maxLimitReached && <p className="max-limit-message">Can't add more past 5!</p>}
        </div>
  
        {/* Drug Information Display */}
        {drugInfo && (
          <div className="drug-info-box">
            <h3>Drug Information</h3>
            {loadingInfo ? <p className="loading-message">Loading...</p> : <p>{drugInfo}</p>}
          </div>
        )}
      </div>
  
      {/* Right Side: Fetch Interactions and Display Results */}
      <div className="right-container">
        <button
          type="button"
          className="fetch-interactions-button"
          onClick={handleFetchInteractions}
        >
          Fetch Symptoms & Interactions
        </button>
  
        {interactionData && (
          <div className="interaction-data-box">
            <h3>Drug Symptoms & Interaction</h3>
            {loadingInteractions ? (
              <p className="loading-message">Loading...</p>
            ) : (
              <div dangerouslySetInnerHTML={{ __html: interactionData }} />
            )}
          </div>
        )}
      </div>
    </div>
  );
  
};

export default AutocompleteInput;
