import React, { Component } from 'react';

class App extends Component {
    state = {
        data: []
    };
    

    componentDidMount() {
        const url = "https://en.wikipedia.org/w/api.php?action=opensearch&search=Seona+Dancing&format=json&origin=*&limit=1";

        fetch(url)
            .then(result => result.json())
            .then(result => {
                this.setState({
                    data: result
                })
            });
    }


    componentDidMount() {
        axios.get('http://127.0.0.1:105/home/dor/Desktop/DolevApp/server/textJSON.json') 
        .then(res => {
        this.setState({ data: res.data });  
        });
    }
    
   
    render() {
        const { data } = this.state;

        const result = data.map((entry, index) => {
            console.log(entry);
            return <li key={index}>{entry}</li>;
        });

        return <div className="container"><ul>{result}</ul></div>;
    }
}

export default App;