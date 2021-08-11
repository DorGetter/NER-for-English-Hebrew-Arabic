import React, { Component } from 'react';
import styled from 'styled-components';
import './App.css';
import { withStyles, makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import { yellow } from '@material-ui/core/colors';
import Tippy from '@tippy.js/react';
import 'tippy.js/dist/tippy.css';
import { Form, TextArea } from 'semantic-ui-react'


document.body.style.background = "#bfd6f6";
const BTN = styled.button`
  background-color: gray;
  display: inline-block;
  font-size: 1em;
  border: 2px solid whitesmoke;
  border-radius: 3px;
  display: block;
  margin-right: 100px;
  width:100px ;
  height: 40px;
  background-color: black;
  box-shadow: 40px;
  color: whitesmoke;

`;
const Title = styled.h1`
  font-size: 1.5em;
  text-align: left;
  padding-right:3em;
  color: white;
`;
const Wrapper = styled.section`
  background:   black;
  padding: 2em;
  box-shadow: 60px;
  padding-left: 10px;
  padding-right: 10px;
`;

const Input = styled.input`
  padding: 0.5em;
  margin: 0.5em;
  color: black;
  background: white ;
  border: none;
  border-radius: 3px;
  width: 1135px;
  margin-right: 670px;
  height: 100px;
  font-weight  : bold;
`;

const insert = styled.text`
text-align: left;
`;


const StringContent = () => (
    <Tippy content="Hello">
      <button>My button</button>
    </Tippy>
  );
   
  const JSXContent = () => (
    <Tippy content={<span>Tooltip</span>}>
      <button>My button</button>
    </Tippy>
  );

const StyledTableCell = withStyles((theme) => ({
    head: {
        backgroundColor: theme.palette.common.black,
        color: theme.palette.common.white,
    },
    body: {
        fontSize: 14,
    },
}))(TableCell);


const StyledTableRow = withStyles((theme) => ({
    root: {
        '&:nth-of-type(odd)': {
            backgroundColor: theme.palette.action.hover,
        },
    },
}))(TableRow);

const useStyles = makeStyles({
    table: {
        minWidth: 700,
    },
});



///////////////// functions /////////////////////

function entityBreaking(val){
    const entities = val.split('^');
    const length = entities.length;
    if(length == 1){
        if(val == 'O'){
            return val;
        }else{
            return entities[0].substring(2);
            }
    }else{
        var lastWord = entities[length-1];
        if(lastWord != 'O'){
            var singelton = lastWord.substring(2);
            return singelton;
        }else{
            return lastWord;
        }
    }
}



function fullEntity_eng(val){
    if(val != 'O'){
        switch (val) {
            case 'ang':
                return val+" (language)";
                break;
            case 'duc':
                return val+" (product)";
                break;
            case 'eve':
                return val+" (event)";
                break;
            case 'Fac':
                return val+" (facility)";
                break;
            case 'Gpe':
                return val+" (geo-political entity)";
                break;
            case 'Loc':
                return val+" (location)";
                break;
            case 'Org':
                return val+" (organization)";
                break;
            case 'per':
                return val+" (person)";
                break;
            case 'Woa':
                return val+" (work-of-art)";
                break;
        }
    }else{
        return val;
    }
}
function fullEntity_he(val){
    if(val != 'O'){
        switch (val) {
            case 'ANG':
                return val+" (language)";
                break;
            case 'DUC':
                return val+" (product)";
                break;
            case 'EVE':
                return val+" (event)";
                break;
            case 'FAC':
                return val+" (facility)";
                break;
            case 'GPE':
                return val+" (geo-political entity)";
                break;
            case 'LOC':
                return val+" (location)";
                break;
            case 'ORG':
                return val+" (organization)";
                break;
            case 'PER':
                return val+" (person)";
                break;
            case 'WOA':
                return val+" (work-of-art)";
                break;
        }
    }else{
        return val;
    }
}
function fullEntity_ar(val) {
    if(val != 'O'){
        switch (val) {
            case 'LOC':
                return val+" (location)";
                break;
            case 'PERS':
                return val+" (person)";
                break;
            case 'ORG':
                return val+" (organization)";
                break;
            case 'MISC':
                return val+" (isn't PERSON,ORG,or LOC)";
                break;
            }
    }else{
        return val;
    }
} 


///////////////// functions /////////////////////

class App extends Component {
    state = {
        characters: [],
        valueInput: "",
        data: [],
        data2: [],
        colors: [],
        modeldata: [],
        flag: false,
        print: [],
        printScores: [],
        Num_of_entities: [],
        table :[],
        language : '',
        title :[<Title>Welcome to Name Entity Recognition</Title>],
        textBox_align : "LTR",
        textBox_placeholder: "Enter Sentence to Bert",
        submit: "Submit",
        result: "Result"
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
        var num_entities = 0;
        var sentesce = "";
        var print_scores = []

        for (let i = 0; i < data.length; i++) {
            sentesce += data[i]
        }


        words = sentesce.split(" ");
        for (let i = 0; i < m.length; i++) {
            var Type = entityBreaking(m[i].entity)
            
            if(this.state.language === '1'){colors2[i] = this.entity2colorEng(Type)}
            if(this.state.language === '2'){colors2[i] = this.entity2colorHeb(Type)}
            if(this.state.language === '3'){colors2[i] = this.entity2colorAr(Type)}  
            
        }

        
        this.setState({ colors: [...colors2] })
        this.setState({ data2: [...words] })
        
        let indents = [];
        for (var i = 0; i < this.state.data2.length; i++) {
            if (m[i] != null && entityBreaking(m[i].entity) != 'O') {
                num_entities += 1;
            }
        }

        this.setState({Num_of_entities: num_entities});

        indents.push(<li><h>
            <mark style={{background: yellow, padding: '0.45em 0.6em', lineHeight: '1', borderRadius: '0.35em' }}>Entities<span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }}>Num : {num_entities}</span></mark>
         </h></li>);

        // TEXTLABLE + TABLE //  
        print_scores.push(
            <h1> Probabilities:  </h1>);
        for (var i = 0; i < this.state.data2.length; i++) {
            if (m[i] != null && entityBreaking(m[i].entity) != 'O') {
                console.log(this.state.colors[i]);

                // table rendering // 
                print_scores.push(<ul><mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                    {this.state.data2[i] + "\t" + (m[i].score * 100).toFixed(3) + "\n"}
                    <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }} key>{entityBreaking(m[i].entity)}</span>
                </mark></ul>);

                // text rendering // 
                indents.push(<Tippy content = {(m[i].score * 100).toFixed(3)} ><mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                    {this.state.data2[i]}
                </mark></Tippy> );
            }
            else indents.push(this.state.data2[i] + " ");
        }

        indents.push(<li><h>
            <mark style={{background: yellow, padding: '0.45em 0.6em', lineHeight: '1', borderRadius: '0.35em' }}>LEGEND<span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }}></span></mark>
         </h></li>);
        let list_dup = []


        // LEGEND En// 
        if (this.state.language==='1'){
        for (var i = 0; i < m.length; i++) {
            if (m[i] != null && m[i].entity != 'O') {
                if (!list_dup.includes(fullEntity_eng(entityBreaking(m[i].entity)))){
                    list_dup.push(fullEntity_eng(entityBreaking(m[i].entity)))
                    indents.push(<mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                    {fullEntity_eng(entityBreaking(m[i].entity))}
                    <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }} key></span>
                </mark>);
            }
        }
        }}
        // LEGEND He // 
        if (this.state.language==='2'){
            for (var i = 0; i < m.length; i++) {
                if (m[i] != null && entityBreaking(m[i].entity) != 'O') {
                    if (!list_dup.includes(fullEntity_he(entityBreaking(m[i].entity)))){
                    list_dup.push(fullEntity_he(entityBreaking(m[i].entity)))
                    indents.push(<mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                        {fullEntity_he(entityBreaking(m[i].entity))}
                        <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }} key></span>
                    </mark>);
                }
            }
            } 
        }
        // LEGEND Ar // 
        if (this.state.language==='3'){
            for (var i = 0; i < m.length; i++) {
                if (m[i] != null && entityBreaking(m[i].entity) != 'O') {
                    if (!list_dup.includes(fullEntity_ar(entityBreaking(m[i].entity)))){
                    list_dup.push(fullEntity_ar(entityBreaking(m[i].entity)))
                    indents.push(<mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                        {fullEntity_ar(entityBreaking(m[i].entity))}
                        <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }} key></span>
                    </mark>);
                }
            }
            } 
        }
        this.setState({ print: [...indents] });
        this.setState({ printScores: [...print_scores] });
        this.create_table();
    }

    create_table(){
         var table2 = []
        table2.push(<TableContainer component={Paper}  align= "justify" padding ="2em">
            <Table  aria-label="customized table" fontWeight = "bold">
                <TableHead  backgroundColor ="blue">
                    <TableRow>
                        <StyledTableCell style = {{align :"justify",fontWeight :"bold"}} >Words</StyledTableCell>
                        <StyledTableCell style = {{align :"justify",fontWeight :"bold"}}> Probabilities</StyledTableCell>
                        <StyledTableCell style = {{align :"justify",fontWeight :"bold"}}>Entity</StyledTableCell>
                    </TableRow>
                </TableHead>
            {this.state.modeldata.map((row,i) =>(
                 <StyledTableRow key={row.word}>
                 <StyledTableCell style = {{align :"justify",fontWeight :"bold" , backgroundColor : this.state.colors[i]}} component="th" scope="row">
                   {row.word}
                 </StyledTableCell>
                 <StyledTableCell style = {{align :"justify" ,fontWeight :"bold" , backgroundColor : this.state.colors[i]}}>{(row.score * 100).toFixed(3)}</StyledTableCell>
                 <StyledTableCell style = {{align :"justify" ,fontWeight :"bold" , backgroundColor : this.state.colors[i]}}>{entityBreaking(row.entity)}</StyledTableCell>
               </StyledTableRow>
             ))}
            </Table>
        </TableContainer>);
        this.setState({ table: [...table2] });

    }
    


    entity2colorEng(val) {

        switch (val) {

            case 'ang':
                return "orange";
                break;
            case 'geo':
                return "orange";
                break;
            case 'per':
                return "#18d596";
                break;
            case 'gpe':    
                return "green"
                break;
            case 'org':
                return "red";
                break;
            case 'tim':
                return "#55f1d5"
                break;
            case 'nat':
                return "#cf6eea"
                break;
            case 'art':
                return "#d56c71"
                break;
            case 'eve':
                return "#c2d56c"
                break;      
            case 'O': 
            case 'PAD':
                return "";
                break;    
            default:
                return 'white';
            
        }
    }

    entity2colorHeb(val){
    switch (val) {
        case 'ANG':
            return "orange";
            break;
        case 'DUC':
            return "red";
            break;
        case 'EVE':
            return "blue";
            break;
        case 'FAC':
            return "brown";
            break;
        case 'GPE':
            return "purple";
            break;
        case 'LOC':
            return "#ede0f9";
            break;
        case 'ORG':
            return "green";
            break;
        case 'PER':
            return "#18d596";
            break;
        case 'WOA':
            return "#07bcf1";
            break;
        case 'O':
        case 'PAD':
            return "";
            break;
        default:
            return 'white';
    }
}
   


    entity2colorAr(val) {

        switch (val) {

            case 'LOC':
                return "orange";
                break;
            case 'PERS':
                return "#18d596";
                break;
            case 'ORG':    
                return "#07bcf1"
                break;
            case 'MISC':
                return "red";
                break;      
            case 'O': 
            case 'PAD':
                return "";
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
        if (this.state.language === "1"){
            fetch('http://127.0.0.1:105/new_code', requestOptions)
                .then(response => response.text())
        }

        if (this.state.language === "2"){
            fetch('http://127.0.0.1:106/new_code', requestOptions)
                .then(response => response.text())
        }

        if (this.state.language === "3") {
            fetch('http://127.0.0.1:107/new_code', requestOptions)
                .then(response => response.text())
        }

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


    handleLang = (c)  =>{

        this.setState({language : c.target.value})
        this.title(c.target.value)
    }

    
    LangSelected = () => {
        let la = "" 
        if (this.state.language === '0'){
            la = "Please Choose Language";
        }
        if (this.state.language === "1"){
            la = "English Selected";
        }
        if (this.state.language === "2"){
            la = "Hebraw Selected";
        }
        if (this.state.language === "3"){
            la = "Arabic Selected"
        }
        return(
            la
        )
    }



    title = (lan) => {

        let t = []
        let t_box_al = ""
        let t_box_pl = ""
        let sub = ""
        let res = ""
        if (lan == 0 || lan == 1){
            t.push(<Title>Welcome to Name Entity Recognition</Title>)
            t_box_al = "LTR"
            t_box_pl = "Enter sentence to Bert"
            sub = "Submit"
            res = "Result"
        }

        else if(lan == 2){
            t.push(<Title style={{textAlign:"right"}}>ברוכים הבאים לזיהוי ישיות בשם</Title>)
            t_box_al = "RTL"
            t_box_pl = "הכנס משפט לברט"
            sub = "שלח"
            res = "תוֹצָאָה"
        }

        else if (lan == 3){
            t.push(<Title style={{textAlign:"right"}}>مرحبًا بكم في التعرف على الكيانات المسماه</Title>)
            t_box_al = "RTL"
            t_box_pl = "أدخل جملة لبيرت"
            sub = "يقدم"
            res = "نتيجة"
        }

        this.setState({title: [...t] })
        this.setState({textBox_align: t_box_al})
        this.setState({textBox_placeholder: t_box_pl})
        this.setState({submit: sub})
        this.setState({result: res})

    }




    render() {
        const {print} = this.state;
        const {table} = this.state;
        const {language} = this.state;
        const {title} = this.state;
        
        return (


            <div className="App">  
                <Wrapper><img src={require('./Bert2.png')} align = "0.5em" height="120em"   justifyContent = 'right'   align = 'right'/> 
                <div>
                <img src={require('./Globe.png')} height="40em" /> 
                        <select  onChange={(e)=>{this.handleLang(e);}}>

                            <option value = '0'>choose lang</option> 
                            <option value = '1'>English</option>
                            <option value = '2'>Hebrew</option>
                            <option value = '3'>Arabic</option>
                        
                        </select>
                        <span style = {{color: "white" }} > {this.LangSelected()}</span>

                    </div>

                    <div> {title}  </div>
                </Wrapper>                
                <Form> 
                    <textarea dir={this.state.textBox_align} aria-expanded = 'true' rows='20' cols='200' itemScope='40em' placeholder={this.state.textBox_placeholder} value={this.state.value} onChange={(e) => {this.handleChange(e)}} />       
                </Form>
                <BTN dir={this.state.textBox_align} onClick={this.sendHttp}> {this.state.submit} </BTN>
                <BTN dir={this.state.textBox_align} onClick={this.getData} type="submit">{this.state.result}</BTN>
                
                <div dir={this.state.textBox_align} > {print}   </div>
                {/* <div> {language}</div> */}
                <div style= {{ display :"block", padding : "1.5em"}}>  {table}   </div>
               
                </div>
        );
    }
}
export default App;