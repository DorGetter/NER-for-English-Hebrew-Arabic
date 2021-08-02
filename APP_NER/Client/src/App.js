import React, { Component } from 'react';

// import {Keywords} from 'react-marker'
class App extends Component {
    state = {
        characters: [],
        valueInput: "",
        data: [],
        data2: [],
        colors: [],
        modeldata: [],
        flag: false,
        print : []
    };

    removeCharacter = index => {
        const { characters } = this.state;

        this.setState({
            characters: characters.filter((character, i) => {
                return i !== index;
            })
        });
    }

    handleSubmit = character => {
        this.setState({ characters: [...this.state.characters, character] });
    }




    dorgetter = () => {
        let m = [...this.state.modeldata];
        let data = [...this.state.data];
        var words = []
        var colors2 = []
        var sentesce = "";
        for (let i = 0; i < data.length; i++) {
            sentesce += data[i]
        }
        words = sentesce.split(" ");
        for (let i = 0; i < m.length; i++) {
            var Type = m[i].entity
            var score = m[i].score
            var color = this.entity2color(Type)
            colors2[i] = color
        }
        this.setState({ colors: [...colors2] })
        this.setState({ data2: [...words] })
        let indents = [];
        for (var i = 0; i < this.state.data2.length; i++) {
            if(m[i].entity != 'O' && m[i].entity != null)
                indents.push(<span style={{ background: this.state.colors[i] }} key>{this.state.data2[i]+" "+m[i].entity+ " "}</span>);
            else
                indents.push(<span style={{ background: this.state.colors[i] }} key>{this.state.data2[i]+" "}</span>);
           
          }
        this.setState({print : [...indents]})
    }



    entity2color(val) {

        switch (val) {
            case 'B-geo':
                return "orange";
                break;
            case 'O':
                return ""
                break
            case 'B-per':
                return "purple";
                break;
            case 'I-geo':
                return "yellow";
                break;
            case 'B-gpe':
                return "green"
                break;
            case 'I-org':
                return "red";
                break;
            case 'B-tim':
                return "blue"
                break;
            default:
                return 'white';
        }
    }

    sendHttp = () => {
        const requestOptions = {
            method: 'POST',
            mode: 'no-cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: this.state.valueInput })
        };

        fetch('http://127.0.0.1:105/new_code', requestOptions)
            .then(response => response.text())
    }

    getData = () => {
        var text = require('./RawText.json'); //(with path)
        var jsontext = require('./textJSON.json'); //(with path)
        this.setState({ data: text })
        this.setState({ modeldata: jsontext })
        this.dorgetter()

    }

    handleChange = (c) => {
        this.setState({ valueInput: c.target.value })
    }

  
    render() {
     
        const { print } = this.state;
        
        return (
            <div className="container">
                <h1>Bert model</h1>
                
                <input type="text" id="fname" name="fname" value={this.state.valueInput} onChange={(e) => { this.handleChange(e) }} />
                <button onClick={this.sendHttp}>submit</button>
                <button onClick={this.getData}>print result</button>

                <div>
                    {print}
                </div>
                

            </div>

        );
    }
}

export default App;